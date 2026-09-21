"""Headless installer core: download Volumes, verify, assemble the Installation.

Given a Payload manifest, a download base URL and a destination directory,
this module downloads each Volume (resumable via HTTP Range, honoring the
system proxy), verifies it against the manifest's size and SHA-256 hash,
extracts it into the destination, and finally checks the total extracted
file count. No GUI lives here — the Setup wizard is a thin wrapper.

Pure stdlib so the frozen Installer stays small.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

MANIFEST_SCHEMA = 1
CHUNK_SIZE = 1024 * 1024
# Disk headroom on top of (uncompressed total + largest Volume) to survive
# filesystem overhead during extraction.
DISK_SLACK_RATIO = 1.05
# Written into the destination while an install is in progress; lets a retry
# recognise its own partial output (and is removed on success).
INSTALL_MARKER = ".eht_analytica_install"


class InstallError(Exception):
    """Raised for any download/verification/assembly failure."""


class InstallCancelled(InstallError):
    """Raised (typically from a progress callback) to abort the install.

    Bypasses the per-Volume download retry loop: cancellation is intentional.
    """


@dataclass(frozen=True)
class VolumeSpec:
    index: int
    filename: str
    size_bytes: int
    sha256: str
    file_count: int


@dataclass(frozen=True)
class Manifest:
    tag: str
    total_bytes: int
    total_files: int
    volumes: tuple[VolumeSpec, ...]

    @classmethod
    def from_dict(cls, data: dict) -> "Manifest":
        if data.get("schema") != MANIFEST_SCHEMA:
            raise InstallError(
                f"Unsupported manifest schema: {data.get('schema')!r} "
                f"(expected {MANIFEST_SCHEMA})"
            )
        try:
            volumes = tuple(
                VolumeSpec(
                    index=v["index"],
                    filename=v["filename"],
                    size_bytes=v["size_bytes"],
                    sha256=v["sha256"],
                    file_count=v["file_count"],
                )
                for v in data["volumes"]
            )
            return cls(
                tag=data["tag"],
                total_bytes=data["total_bytes"],
                total_files=data["total_files"],
                volumes=volumes,
            )
        except KeyError as e:
            raise InstallError(f"Manifest is missing field: {e}") from e


def load_manifest(path: Path) -> Manifest:
    import json
    return Manifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _free_bytes(path: Path) -> int:
    """Free space on the filesystem containing ``path`` (or nearest ancestor)."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def _prepare_destination(dest_dir: Path) -> None:
    """Replace an existing Installation wholesale; refuse foreign directories.

    A previous Installation is recognised by its executable or by the
    in-progress marker left by an interrupted earlier attempt, so retries
    can continue. Any other non-empty directory is refused rather than
    wiped — the user must pick an empty or new location.
    """
    marker = dest_dir / INSTALL_MARKER
    if dest_dir.exists():
        entries = list(dest_dir.iterdir())
        ours = marker.exists() or (dest_dir / "EHT_Analytica.exe").exists()
        if entries and not ours:
            raise InstallError(
                f"Destination {dest_dir} already exists and is not an EHT "
                f"Analytica installation. Choose an empty or new directory."
            )
        for child in entries:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    dest_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text("", encoding="utf-8")


def _default_opener() -> urllib.request.OpenerDirector:
    """Proxy-aware opener: uses system proxy settings (Windows registry/env)."""
    return urllib.request.build_opener(
        urllib.request.ProxyHandler(urllib.request.getproxies())
    )


ProgressCB = Callable[[str, dict], None]


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def download_volume(url: str, dest: Path, expected_size: int, expected_sha256: str,
                    progress: Optional[Callable[[int, int], None]] = None,
                    resume: bool = True, max_attempts: int = 3,
                    timeout: float = 60.0,
                    opener: Optional[urllib.request.OpenerDirector] = None
                    ) -> Path:
    """Download ``url`` to ``dest`` with Range-resume and hash verification.

    A file that already exists with the right size and hash is returned
    immediately (cheap retries). Retries up to ``max_attempts`` times: a
    partial file smaller than the expected size is resumed on the next
    attempt; a completed file whose hash mismatches is deleted and
    re-fetched from scratch.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if (dest.exists() and dest.stat().st_size == expected_size
            and _sha256_of_file(dest) == expected_sha256):
        return dest
    op = opener or _default_opener()
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            _download_once(url, dest, expected_size, expected_sha256,
                           progress, resume, timeout, op)
            return dest
        except InstallCancelled:
            raise
        except InstallError as e:
            last_error = e
    raise InstallError(
        f"Failed to download {url} after {max_attempts} attempts: {last_error}"
    )


def _download_once(url: str, dest: Path, expected_size: int, expected_sha256: str,
                   progress, resume: bool, timeout: float,
                   opener: urllib.request.OpenerDirector) -> None:
    headers = {"User-Agent": "EHT-Analytica-Setup"}
    partial = dest.stat().st_size if dest.exists() else 0
    if resume and 0 < partial < expected_size:
        headers["Range"] = f"bytes={partial}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        resp = opener.open(req, timeout=timeout)
        with resp:
            if resp.status == 206 and "Range" in headers:
                mode, done = "ab", partial
            else:
                # Full response: server ignored the Range (or fresh start)
                mode, done = "wb", 0
            with open(dest, mode) as out:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, expected_size)
    except urllib.error.HTTPError as e:
        raise InstallError(f"HTTP {e.code} fetching {url}") from e
    except urllib.error.URLError as e:
        raise InstallError(f"Cannot reach {url}: {e.reason}") from e
    except OSError as e:
        raise InstallError(f"I/O error downloading {url}: {e}") from e

    actual = dest.stat().st_size
    if actual != expected_size:
        # Keep the partial file if smaller (resumable), drop if larger
        if actual > expected_size:
            dest.unlink(missing_ok=True)
        raise InstallError(
            f"Size mismatch for {dest.name}: got {actual}, expected {expected_size}"
        )
    digest = _sha256_of_file(dest)
    if digest != expected_sha256:
        dest.unlink(missing_ok=True)  # corrupt: not safe to resume
        raise InstallError(
            f"SHA-256 mismatch for {dest.name}: got {digest}, "
            f"expected {expected_sha256}"
        )


def install_from_manifest(manifest: Manifest, base_url: str, dest_dir: Path,
                          progress: Optional[ProgressCB] = None,
                          max_attempts: int = 3,
                          keep_volumes: bool = False,
                          opener: Optional[urllib.request.OpenerDirector] = None
                          ) -> Path:
    """Download all Volumes from ``base_url`` and assemble ``dest_dir``.

    Fires progress events: ``download`` (per-Volume byte counters),
    ``extract`` (per-Volume), ``done``. Raises InstallError on any failure
    (disk space, size/hash mismatch, extraction error, file-count mismatch).
    """
    dest_dir = Path(dest_dir)
    total_volumes = len(manifest.volumes)

    if not manifest.volumes:
        raise InstallError("Manifest contains no volumes")

    largest = max(v.size_bytes for v in manifest.volumes)
    required = int((manifest.total_bytes + largest) * DISK_SLACK_RATIO)
    free = _free_bytes(dest_dir)
    if free < required:
        raise InstallError(
            f"Not enough disk space: need ~{required / (1024**3):.1f} GiB, "
            f"only {free / (1024**3):.1f} GiB free at {dest_dir}"
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = dest_dir.parent / f".{dest_dir.name}.setup-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    _prepare_destination(dest_dir)

    extracted_files = 0
    success = False
    try:
        for i, vol in enumerate(manifest.volumes, start=1):
            url = base_url.rstrip("/") + "/" + vol.filename
            zip_path = cache_dir / vol.filename

            if progress:
                progress("download", {
                    "volume": vol.filename, "index": i, "count": total_volumes,
                    "bytes_done": 0, "bytes_total": vol.size_bytes,
                })

            def byte_progress(done: int, total: int, _v=vol, _i=i) -> None:
                if progress:
                    progress("download", {
                        "volume": _v.filename, "index": _i, "count": total_volumes,
                        "bytes_done": done, "bytes_total": total,
                    })

            download_volume(url, zip_path, vol.size_bytes, vol.sha256,
                            progress=byte_progress, max_attempts=max_attempts,
                            opener=opener)

            if progress:
                progress("extract", {
                    "volume": vol.filename, "index": i, "count": total_volumes,
                })
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    members = [m for m in zf.namelist() if not m.endswith("/")]
                    zf.extractall(dest_dir)
            except (zipfile.BadZipFile, OSError) as e:
                raise InstallError(f"Extraction failed for {vol.filename}: {e}") from e
            if len(members) != vol.file_count:
                raise InstallError(
                    f"Volume {vol.filename} contained {len(members)} files, "
                    f"manifest expected {vol.file_count}"
                )
            extracted_files += len(members)

            if not keep_volumes:
                zip_path.unlink(missing_ok=True)
        success = True
    finally:
        # Keep the cache on failure/cancellation so a retry can resume
        # partially downloaded Volumes instead of starting over.
        if success and not keep_volumes:
            shutil.rmtree(cache_dir, ignore_errors=True)

    if extracted_files != manifest.total_files:
        raise InstallError(
            f"Assembled {extracted_files} files, manifest expected "
            f"{manifest.total_files}"
        )

    (dest_dir / INSTALL_MARKER).unlink(missing_ok=True)
    if progress:
        progress("done", {"count": total_volumes})
    return dest_dir


def reassemble_locally(volume_dir: Path, manifest: Manifest,
                       dest_dir: Path) -> Path:
    """Assemble an Installation from Volumes already on disk (verification)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for vol in manifest.volumes:
        src = Path(volume_dir) / vol.filename
        if not src.exists():
            raise InstallError(f"Missing volume: {src}")
        digest = _sha256_of_file(src)
        if digest != vol.sha256:
            raise InstallError(f"SHA-256 mismatch for {vol.filename}")
        with zipfile.ZipFile(src) as zf:
            zf.extractall(dest_dir)
    return dest_dir
