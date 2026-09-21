"""Tests for the inference-sample collector (functions.inference_collector)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from functions.inference_collector import (
    delete_sample,
    get_all_samples_sorted,
    mark_checked,
    save_sample,
    update_coords_and_heatmap,
)


@pytest.fixture()
def roi() -> np.ndarray:
    """A deterministic 64x48 grayscale ROI (H=64, W=48)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(64, 48), dtype=np.uint8)


@pytest.fixture()
def coords() -> np.ndarray:
    return np.array([[10.0, 20.0], [38.0, 44.0]])


def _save(roi, coords, tmp_path, **overrides) -> dict:
    kwargs = dict(
        roi_gray=roi,
        coords=coords,
        model_name="unet_v3",
        target_size=256,
        recording_dir="250904_rEHT",
        recording="EHT-1_d2_0000",
        roi_name="ROI1",
        frame_idx=0,
        data_dir=str(tmp_path / "inferences"),
    )
    kwargs.update(overrides)
    return save_sample(**kwargs)


def test_save_sample_writes_image_and_registry(roi, coords, tmp_path):
    entry = _save(roi, coords, tmp_path)

    data_dir = tmp_path / "inferences"
    roi_file = data_dir / f"{entry['id']}_ROI.jpg"
    assert roi_file.exists()
    assert roi_file.stat().st_size > 0

    registry = json.loads((data_dir / "trainsets.json").read_text(encoding="utf-8"))
    assert len(registry["samples"]) == 1
    assert registry["samples"][0]["id"] == entry["id"]
    assert registry["samples"][0]["coords"] == coords.tolist()
    assert registry["samples"][0]["checked"] is False


def test_save_sample_replaces_same_id(roi, coords, tmp_path):
    _save(roi, coords, tmp_path)
    # Same identity: frames 0 and 29 fall into the same bucket (idx // 30)
    _save(roi, coords, tmp_path, frame_idx=29)

    samples = get_all_samples_sorted(str(tmp_path / "inferences"))
    assert len(samples) == 1
    assert samples[0]["frame_idx"] == 29
    assert len(list((tmp_path / "inferences").glob("*_ROI.jpg"))) == 1


def test_data_dir_is_required(roi, coords):
    with pytest.raises(ValueError, match="data_dir"):
        save_sample(
            roi_gray=roi, coords=coords, model_name="unet_v3", target_size=256,
            recording_dir="d", recording="r", roi_name="ROI1", frame_idx=0,
        )


def test_update_coords_and_heatmap_roundtrip(roi, coords, tmp_path):
    entry = _save(roi, coords, tmp_path)
    data_dir = tmp_path / "inferences"

    new_coords = [[12.0, 22.0], [36.0, 42.0]]
    assert update_coords_and_heatmap(entry["id"], new_coords, data_dir=str(data_dir))

    samples = get_all_samples_sorted(str(data_dir))
    assert samples[0]["coords"] == new_coords
    heatmap = data_dir / f"{entry['id']}_heatmap.jpg"
    assert heatmap.exists()


def test_mark_checked_and_delete(roi, coords, tmp_path):
    entry = _save(roi, coords, tmp_path)
    data_dir = str(tmp_path / "inferences")

    assert mark_checked(entry["id"], data_dir=data_dir)
    assert get_all_samples_sorted(data_dir)[0]["checked"] is True

    assert delete_sample(entry["id"], data_dir=data_dir)
    assert get_all_samples_sorted(data_dir) == []
    assert not (tmp_path / "inferences" / f"{entry['id']}_ROI.jpg").exists()
