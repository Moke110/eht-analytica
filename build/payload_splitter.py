"""Split a packaged onedir tree into Payload Volumes plus a manifest.

A Volume is a self-contained zip (under the per-asset limit of the release
host, by default ~1.77 GiB to stay safely below GitHub's 2 GiB cap); each
Volume extracts into the same root. Files are packed largest-first (with a
relative-path tie-break) so the split is deterministic: the same input tree
always yields the same Volume file lists.

The manifest records each Volume's filename, byte size, SHA-256 hash and
file count. It is the contract between the release orchestration (which
produces it) and the Installer (which consumes it).

Usage:
    python build/payload_splitter.py <onedir> <out_dir> --tag v0.4.0
        [--max-volume-mb 1900] [--verify]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

APP_NAME = "EHT_Analytica"
MANIFEST_SCHEMA = 1
DEFAULT_MAX_VOLUME_MB = 1900
# Conservative zip overhead estimates used when planning Volume contents:
# per-entry local+central headers plus filename, plus the end-of-archive
# record. Real archives stay under this for typical trees, so the runtime
# size check below almost never trips.
ZIP_PER_FILE_OVERHEAD = 200
ZIP_FIXED_OVERHEAD = 128


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_tree_files(root: Path) -> list[Path]:
    """All files under root, in deterministic (sorted by relative path) order."""
    return sorted((p for p in root.rglob("*") if p.is_file()),
                  key=lambda p: p.relative_to(root).as_posix())


def _plan_volumes(files: list[Path], root: Path,
                  max_volume_bytes: int) -> list[list[Path]]:
    """First-fit-decreasing assignment of files to Volumes.

    Deterministic: files are ordered by size descending, then by relative
    path. Packing uses estimated *archived* size (raw size plus per-entry
    zip overhead) so the produced Volume stays under the limit. Raises if a
    single file cannot fit in one Volume.
    """
    def est(f: Path) -> int:
        return (f.stat().st_size + ZIP_PER_FILE_OVERHEAD
                + len(f.relative_to(root).as_posix()))

    ordered = sorted(files, key=lambda p: (-p.stat().st_size,
                                           p.relative_to(root).as_posix()))
    volumes: list[list[Path]] = []
    used = ZIP_FIXED_OVERHEAD
    for f in ordered:
        size = est(f)
        if f.stat().st_size > max_volume_bytes:
            rel = f.relative_to(root).as_posix()
            raise ValueError(
                f"Single file exceeds max Volume size ({f.stat().st_size} > "
                f"{max_volume_bytes} bytes): {rel}. Raise --max-volume-mb or "
                f"repack the dependency."
            )
        if not volumes or used + size > max_volume_bytes:
            volumes.append([f])
            used = ZIP_FIXED_OVERHEAD + size
        else:
            volumes[-1].append(f)
            used += size
    return volumes


def _volume_filename(tag: str, index: int) -> str:
    return f"{APP_NAME}-{tag}-payload-{index:02d}.zip"


def split_payload(onedir: Path, out_dir: Path, tag: str,
                  max_volume_bytes: int = DEFAULT_MAX_VOLUME_MB * 1024 * 1024,
                  progress=print) -> dict:
    """Split ``onedir`` into Volumes under ``out_dir``; return the manifest dict."""
    files = iter_tree_files(onedir)
    if not files:
        raise ValueError(f"No files found under {onedir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    # Stale volumes from a previous split would corrupt the manifest set
    for old in out_dir.glob(f"{APP_NAME}-{tag}-payload-*.zip"):
        old.unlink()

    plan = _plan_volumes(files, onedir, max_volume_bytes)
    volumes_meta = []
    total_bytes = 0
    total_files = 0

    for i, vol_files in enumerate(plan, start=1):
        filename = _volume_filename(tag, i)
        path = out_dir / filename
        # Write archive entries in sorted order for reproducible zips
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for f in sorted(vol_files,
                            key=lambda p: p.relative_to(onedir).as_posix()):
                zf.write(f, f.relative_to(onedir).as_posix())
        size = path.stat().st_size
        if size > max_volume_bytes:
            raise RuntimeError(
                f"Volume {filename} is {size} bytes (> {max_volume_bytes}) "
                f"despite plan; incompressible tree needs a lower --max-volume-mb."
            )
        volumes_meta.append({
            "index": i,
            "filename": filename,
            "size_bytes": size,
            "sha256": sha256_of_file(path),
            "file_count": len(vol_files),
        })
        total_bytes += sum(f.stat().st_size for f in vol_files)
        total_files += len(vol_files)
        progress(f"  Volume {i}/{len(plan)}: {filename} "
                 f"({size / (1024 * 1024):.1f} MB, {len(vol_files)} files)")

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "app_name": APP_NAME,
        "tag": tag,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_bytes": total_bytes,
        "total_files": total_files,
        "volumes": volumes_meta,
    }
    manifest_path = out_dir / "release-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    progress(f"  Manifest: {manifest_path} "
             f"({len(plan)} volumes, {total_files} files, "
             f"{total_bytes / (1024 * 1024):.1f} MB uncompressed)")
    return manifest


def reassemble_volumes(volume_paths: list[Path], dest: Path) -> None:
    """Extract Volumes (in index order) into ``dest`` — used for verification."""
    dest.mkdir(parents=True, exist_ok=True)
    for vp in volume_paths:
        with zipfile.ZipFile(vp, "r") as zf:
            zf.extractall(dest)


def trees_identical(a: Path, b: Path) -> tuple[bool, str]:
    """Byte-for-byte comparison of two directory trees."""
    fa = {p.relative_to(a).as_posix(): p for p in iter_tree_files(a)}
    fb = {p.relative_to(b).as_posix(): p for p in iter_tree_files(b)}
    if set(fa) != set(fb):
        only_a = sorted(set(fa) - set(fb))[:5]
        only_b = sorted(set(fb) - set(fa))[:5]
        return False, f"file sets differ; only in source: {only_a}, only in dest: {only_b}"
    for rel in sorted(fa):
        if fa[rel].read_bytes() != fb[rel].read_bytes():
            return False, f"content differs: {rel}"
    return True, ""


def verify_payload(onedir: Path, out_dir: Path, manifest: dict) -> bool:
    """Reassemble the Volumes into a temp dir and compare against ``onedir``."""
    with tempfile.TemporaryDirectory(prefix="eht_verify_") as tmp:
        dest = Path(tmp) / "reassembled"
        volume_paths = [out_dir / v["filename"] for v in manifest["volumes"]]
        reassemble_volumes(volume_paths, dest)
        ok, why = trees_identical(onedir, dest)
        if not ok:
            print(f"VERIFY FAILED: {why}", file=sys.stderr)
            return False
    print("Verify OK: reassembled tree is byte-for-byte identical to source.")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("onedir", type=Path, help="packaged onedir tree to split")
    parser.add_argument("out_dir", type=Path, help="directory for Volumes + manifest")
    parser.add_argument("--tag", required=True, help="release tag, e.g. v0.4.0")
    parser.add_argument("--max-volume-mb", type=int, default=DEFAULT_MAX_VOLUME_MB,
                        help="max Volume size in MiB (default %(default)s)")
    parser.add_argument("--verify", action="store_true",
                        help="after splitting, reassemble and compare byte-for-byte")
    args = parser.parse_args(argv)

    manifest = split_payload(args.onedir.resolve(), args.out_dir.resolve(),
                             args.tag, args.max_volume_mb * 1024 * 1024)
    if args.verify:
        if not verify_payload(args.onedir.resolve(), args.out_dir.resolve(), manifest):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
