"""Tests for the headless installer core (offline, local files only)."""

from __future__ import annotations

import http.server
import json
import re
import threading
import urllib.request
from pathlib import Path

import pytest

import installer_core
from installer_core import (
    InstallError,
    Manifest,
    download_volume,
    install_from_manifest,
    load_manifest,
)
from payload_splitter import split_payload, trees_identical


@pytest.fixture()
def synthetic_tree(tmp_path):
    root = tmp_path / "onedir"
    layout = {
        "EHT_Analytica.exe": 900,
        "lib/torch/a.dll": 400,
        "lib/torch/b.dll": 350,
        "model/weights.pth": 200,
        "frontend/index.html": 80,
    }
    for rel, size in layout.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(bytes([(i * 7) % 256 for i in range(size)]))
    return root


@pytest.fixture()
def payload(synthetic_tree, tmp_path):
    """Split tree -> (release_dir, manifest obj, manifest dict)."""
    out = tmp_path / "release"
    manifest_dict = split_payload(synthetic_tree, out, tag="v0.0.1-test",
                                  max_volume_bytes=1000, progress=lambda *_: None)
    return out, Manifest.from_dict(manifest_dict), manifest_dict


def test_manifest_roundtrip_validation(payload):
    _, manifest, raw = payload
    assert manifest.tag == "v0.0.1-test"
    assert manifest.total_files == 5
    assert len(manifest.volumes) >= 2
    assert manifest == Manifest.from_dict(raw)


def test_manifest_rejects_unknown_schema():
    with pytest.raises(InstallError, match="schema"):
        Manifest.from_dict({"schema": 99, "volumes": [], "tag": "x",
                            "total_bytes": 0, "total_files": 0})


def test_install_round_trip_from_file_url(payload, tmp_path):
    release_dir, manifest, _ = payload
    dest = tmp_path / "Installation"

    events = []
    install_from_manifest(manifest, release_dir.as_uri(), dest,
                          progress=lambda kind, info: events.append(kind))

    ok, why = trees_identical(tmp_path / "onedir", dest)
    assert ok, why
    assert events[0] == "download"
    assert "extract" in events
    assert events[-1] == "done"
    # Cache dir cleaned up
    assert not (tmp_path / ".Installation.setup-cache").exists()


def test_corrupted_volume_rejected(payload, tmp_path):
    release_dir, manifest, _ = payload
    # Flip a byte in the first volume
    vol = release_dir / manifest.volumes[0].filename
    data = bytearray(vol.read_bytes())
    data[len(data) // 2] ^= 0xFF
    vol.write_bytes(bytes(data))

    with pytest.raises(InstallError, match=manifest.volumes[0].filename):
        install_from_manifest(manifest, release_dir.as_uri(),
                              tmp_path / "Installation", max_attempts=1)
    # Failed installs keep the download cache so a retry can resume
    assert (tmp_path / ".Installation.setup-cache").exists()


def test_install_replaces_existing_installation(payload, tmp_path):
    """An upgrade replaces the Installation wholesale (no stale files)."""
    release_dir, manifest, _ = payload
    dest = tmp_path / "Installation"
    dest.mkdir()
    (dest / "EHT_Analytica.exe").write_bytes(b"old build")
    stale = dest / "lib" / "old_module.dll"
    stale.parent.mkdir()
    stale.write_bytes(b"stale")

    install_from_manifest(manifest, release_dir.as_uri(), dest)

    assert trees_identical(tmp_path / "onedir", dest)[0]
    assert not stale.exists()


def test_install_refuses_foreign_directory(payload, tmp_path):
    """A non-empty, non-Installation destination is never wiped."""
    release_dir, manifest, _ = payload
    dest = tmp_path / "SomeonesFolder"
    dest.mkdir()
    precious = dest / "thesis.docx"
    precious.write_bytes(b"important")

    with pytest.raises(InstallError, match="not an EHT"):
        install_from_manifest(manifest, release_dir.as_uri(), dest)
    assert precious.exists()


def test_install_marker_removed_on_success(payload, tmp_path):
    release_dir, manifest, _ = payload
    dest = tmp_path / "Installation"
    install_from_manifest(manifest, release_dir.as_uri(), dest)
    assert not (dest / installer_core.INSTALL_MARKER).exists()


def test_cancel_during_download_raises_and_keeps_cache(payload, tmp_path):
    from installer_core import InstallCancelled

    release_dir, manifest, _ = payload
    dest = tmp_path / "Installation"

    def cancel_on_first_download(kind, info):
        if kind == "download":
            raise InstallCancelled("cancelled by test")

    with pytest.raises(InstallCancelled):
        install_from_manifest(manifest, release_dir.as_uri(), dest,
                              progress=cancel_on_first_download)
    # Cache survives cancellation so a retry resumes instead of restarting
    assert (tmp_path / ".Installation.setup-cache").exists()


def test_truncated_volume_rejected(payload, tmp_path):
    release_dir, manifest, _ = payload
    vol = release_dir / manifest.volumes[0].filename
    data = vol.read_bytes()
    vol.write_bytes(data[: len(data) - 10])

    with pytest.raises(InstallError):
        install_from_manifest(manifest, release_dir.as_uri(),
                              tmp_path / "Installation", max_attempts=1)


def test_missing_volume_rejected(payload, tmp_path):
    release_dir, manifest, _ = payload
    (release_dir / manifest.volumes[0].filename).unlink()
    with pytest.raises(InstallError):
        install_from_manifest(manifest, release_dir.as_uri(),
                              tmp_path / "Installation", max_attempts=1)


def test_disk_space_check(payload, tmp_path, monkeypatch):
    _, manifest, _ = payload
    monkeypatch.setattr(installer_core, "_free_bytes", lambda p: 100)
    with pytest.raises(InstallError, match="disk space"):
        install_from_manifest(manifest, "file:///unused", tmp_path / "Installation")


def test_file_count_mismatch_detected(payload, tmp_path):
    """A volume zip missing one entry must fail the count check."""
    release_dir, manifest, _ = payload
    import hashlib
    import zipfile
    vol = release_dir / manifest.volumes[0].filename

    # Rewrite the volume without its last member
    with zipfile.ZipFile(vol) as zf:
        names = zf.namelist()
        entries = {n: zf.read(n) for n in names}
    with zipfile.ZipFile(vol, "w") as zf:
        for n in names[:-1]:
            zf.writestr(n, entries[n])
    # Re-sync the manifest entry with the rewritten volume
    new_size = vol.stat().st_size
    new_sha = hashlib.sha256(vol.read_bytes()).hexdigest()
    volumes = list(manifest.volumes)
    v0 = volumes[0]
    volumes[0] = type(v0)(index=v0.index, filename=v0.filename,
                          size_bytes=new_size, sha256=new_sha,
                          file_count=v0.file_count)  # count now stale
    stale = Manifest(tag=manifest.tag, total_bytes=manifest.total_bytes,
                     total_files=manifest.total_files, volumes=tuple(volumes))

    with pytest.raises(InstallError, match="manifest expected"):
        install_from_manifest(stale, release_dir.as_uri(),
                              tmp_path / "Installation", max_attempts=1)


def test_load_manifest_from_file(payload, tmp_path):
    release_dir, _, raw = payload
    m = load_manifest(release_dir / "release-manifest.json")
    assert m.total_files == raw["total_files"]


# ---------------------------------------------------------------------------
# Range-resume over a local HTTP server
# ---------------------------------------------------------------------------

class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Minimal GET server with HTTP Range support, recording requests."""

    protocol_version = "HTTP/1.1"

    def do_GET(self):  # noqa: N802 (http.server naming)
        import os
        path = Path(self.server.directory) / urllib.parse.unquote(
            self.path.lstrip("/"))
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        total = len(data)
        rng = self.headers.get("Range")
        self.server.requests.append({"path": self.path, "range": rng})
        if rng:
            start = int(re.match(r"bytes=(\d+)-", rng).group(1))
            chunk = data[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{total-1}/{total}")
        else:
            chunk = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(chunk)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(chunk)

    def log_message(self, *args):
        pass


@pytest.fixture()
def http_server(tmp_path):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    server.directory = str(tmp_path)
    server.requests = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", server
    server.shutdown()


def _no_proxy_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def test_download_resumes_from_partial(http_server, tmp_path):
    base, server = http_server
    src = tmp_path / "vol.zip"
    src.write_bytes(bytes([(i * 13) % 256 for i in range(5000)]))
    import hashlib
    expected_sha = hashlib.sha256(src.read_bytes()).hexdigest()

    dest = tmp_path / "download" / "vol.zip"
    dest.parent.mkdir()
    # Simulate an interrupted earlier attempt: first 2000 bytes on disk
    dest.write_bytes(src.read_bytes()[:2000])

    seen = []
    download_volume(f"{base}/vol.zip", dest, 5000, expected_sha,
                    opener=_no_proxy_opener(), max_attempts=2)

    assert dest.read_bytes() == src.read_bytes()

    ranges = [r["range"] for r in server.requests if r["path"] == "/vol.zip"]
    assert any(r and r.startswith("bytes=2000-") for r in ranges), ranges


def test_download_retries_on_hash_mismatch(http_server, tmp_path):
    base, server = http_server
    src = tmp_path / "vol.zip"
    src.write_bytes(b"A" * 1000)

    dest = tmp_path / "dl" / "vol.zip"
    with pytest.raises(InstallError, match="SHA-256"):
        download_volume(f"{base}/vol.zip", dest, 1000, "0" * 64,
                        opener=_no_proxy_opener(), max_attempts=2)
    # Both attempts were full downloads (corrupt file was deleted, not resumed)
    vol_requests = [r for r in server.requests if r["path"] == "/vol.zip"]
    assert len(vol_requests) == 2
    assert not dest.exists()
