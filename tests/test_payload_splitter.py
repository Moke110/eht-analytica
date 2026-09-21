"""Tests for the Payload volume splitter and manifest format."""

from __future__ import annotations

import hashlib
import json
import zipfile

import pytest

from payload_splitter import (
    reassemble_volumes,
    split_payload,
    trees_identical,
    _plan_volumes,
    _volume_filename,
)


@pytest.fixture()
def synthetic_tree(tmp_path):
    """A tree with nested dirs and files of varied sizes (bytes)."""
    root = tmp_path / "onedir"
    layout = {
        "EHT_Analytica.exe": 500,
        "lib/torch/big.dll": 300,
        "lib/torch/other.dll": 250,
        "lib/cv2/cv2.dll": 200,
        "model/unet_v3/weights.pth": 150,
        "frontend_dist/index.html": 100,
        "frontend_dist/assets/app.js": 90,
    }
    for rel, size in layout.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(bytes([i % 256 for i in range(size)]))
    return root


def _split(root, out, max_bytes=1000):
    return split_payload(root, out, tag="v0.0.1-test",
                         max_volume_bytes=max_bytes, progress=lambda *_: None)


def test_split_produces_manifest_and_volumes(synthetic_tree, tmp_path):
    out = tmp_path / "release"
    manifest = _split(synthetic_tree, out)

    assert manifest["schema"] == 1
    assert manifest["tag"] == "v0.0.1-test"
    assert manifest["total_files"] == 7
    assert manifest["total_bytes"] == sum(
        f.stat().st_size for f in synthetic_tree.rglob("*") if f.is_file()
    )
    assert len(manifest["volumes"]) >= 2  # 1590 bytes / 1000 per volume

    for v in manifest["volumes"]:
        path = out / v["filename"]
        assert path.exists()
        assert path.stat().st_size == v["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == v["sha256"]
        with zipfile.ZipFile(path) as zf:
            assert len(zf.namelist()) == v["file_count"]

    # Manifest file written next to the volumes
    on_disk = json.loads((out / "release-manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest


def test_round_trip_reassembles_identical_tree(synthetic_tree, tmp_path):
    out = tmp_path / "release"
    manifest = _split(synthetic_tree, out)

    dest = tmp_path / "reassembled"
    reassemble_volumes([out / v["filename"] for v in manifest["volumes"]], dest)
    ok, why = trees_identical(synthetic_tree, dest)
    assert ok, why


def test_each_volume_under_limit_and_complete_coverage(synthetic_tree, tmp_path):
    out = tmp_path / "release"
    manifest = _split(synthetic_tree, out, max_bytes=600)

    seen: set[str] = set()
    for v in manifest["volumes"]:
        assert v["size_bytes"] < 600
        with zipfile.ZipFile(out / v["filename"]) as zf:
            seen.update(zf.namelist())
    expected = {p.relative_to(synthetic_tree).as_posix()
                for p in synthetic_tree.rglob("*") if p.is_file()}
    assert seen == expected  # every file exactly once


def test_split_is_deterministic(synthetic_tree, tmp_path):
    m1 = _split(synthetic_tree, tmp_path / "a")
    m2 = _split(synthetic_tree, tmp_path / "b")
    assert [v["file_count"] for v in m1["volumes"]] == [v["file_count"] for v in m2["volumes"]]
    assert [v["sha256"] for v in m1["volumes"]] == [v["sha256"] for v in m2["volumes"]]


def test_single_file_too_large_raises(tmp_path):
    root = tmp_path / "onedir"
    root.mkdir()
    (root / "huge.dll").write_bytes(b"x" * 2000)
    with pytest.raises(ValueError, match="exceeds max Volume size"):
        _split(root, tmp_path / "out", max_bytes=1000)


def test_empty_tree_raises(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(ValueError, match="No files found"):
        _split(root, tmp_path / "out")


def test_plan_first_fit_decreasing(tmp_path):
    root = tmp_path / "t"
    root.mkdir()
    sizes = [500, 300, 250, 200, 150, 100]
    for i, s in enumerate(sizes):
        (root / f"f{i}.bin").write_bytes(b"a" * s)
    plan = _plan_volumes(sorted(root.iterdir(), key=lambda p: p.name), root, 1400)

    # All files packed exactly once, larger files first overall
    packed = [p.name for vol in plan for p in vol]
    assert sorted(packed) == [f"f{i}.bin" for i in range(6)]
    assert packed == sorted(packed, key=lambda n: -sizes[int(n[1])])

    # Every volume fits the budget (raw bytes + zip overhead estimate)
    from payload_splitter import ZIP_FIXED_OVERHEAD, ZIP_PER_FILE_OVERHEAD
    for vol in plan:
        est = ZIP_FIXED_OVERHEAD + sum(
            p.stat().st_size + ZIP_PER_FILE_OVERHEAD + len(p.name) for p in vol
        )
        assert est <= 1400

    # Biggest file alone would leave room for the 300-byte file;
    # 500+300 both fit -> they share a volume
    assert [p.name for p in plan[0]] == ["f0.bin", "f1.bin"]


def test_volume_filename_format():
    assert _volume_filename("v0.4.0", 1) == "EHT_Analytica-v0.4.0-payload-01.zip"
    assert _volume_filename("v0.4.0", 12) == "EHT_Analytica-v0.4.0-payload-12.zip"


def test_stale_release_artifacts_removed_on_resplit(synthetic_tree, tmp_path):
    out = tmp_path / "release"
    _split(synthetic_tree, out, max_bytes=1000)
    # Artifacts from a previous run/tag, plus a stale Setup exe
    stale_volume = out / _volume_filename("v0.0.1-test", 99)
    stale_other_tag = out / "EHT_Analytica-v9.9.9-payload-01.zip"
    stale_setup = out / "EHT_Analytica-Setup-v9.9.9.exe"
    for p in (stale_volume, stale_other_tag, stale_setup):
        p.write_bytes(b"stale")

    _split(synthetic_tree, out, max_bytes=1000)

    for p in (stale_volume, stale_other_tag, stale_setup):
        assert not p.exists()
