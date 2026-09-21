"""
Inference sample collector — saves ROI images and coordinates during tracking
for later review as training data.

This module is part of the shipped application: the tracker calls
``save_sample`` during inference when the user enables "Save inferences" in
the GUI. The developer-side review tool (in the private training tree)
imports the review functions from here as well, so the ``trainsets.json``
registry format has a single implementation.

Heatmaps are generated later during the review step (from the final
human-verified coordinates).
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


def _resolve_dir(data_dir: Optional[str] = None) -> str:
    """Return data_dir; a data directory is required — there is no default."""
    if data_dir is None:
        raise ValueError("data_dir is required: the collector is location-independent")
    return data_dir


def _trainsets_path(data_dir: Optional[str] = None) -> str:
    return os.path.join(_resolve_dir(data_dir), "trainsets.json")


def _ensure_dir(data_dir: Optional[str] = None) -> str:
    d = _resolve_dir(data_dir)
    os.makedirs(d, exist_ok=True)
    return d


def _load_registry(data_dir: Optional[str] = None) -> dict:
    path = _trainsets_path(data_dir)
    if os.path.isfile(path):
        for attempt in range(3):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                if attempt == 2:
                    raise
                time.sleep(0.1 * (attempt + 1))
    return {"samples": [], "next_id": 0}


def _save_registry(data: dict, data_dir: Optional[str] = None) -> None:
    _ensure_dir(data_dir)
    path = _trainsets_path(data_dir)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)  # atomic rename on Windows & Linux


def save_sample(roi_gray, coords: np.ndarray,
                model_name: str, target_size: int,
                recording_dir: str, recording: str, roi_name: str,
                frame_idx: int,
                data_dir: Optional[str] = None) -> dict:
    """Save one inference sample to the given data directory.

    Parameters
    ----------
    roi_gray : torch.Tensor or np.ndarray
        Raw grayscale ROI (H, W), values 0-255.
    coords : np.ndarray
        Pillar coordinates (num_peaks, 2) in original ROI pixel space.
    model_name : str
        Display name from models.json.
    target_size : int
        Model input size (for traceability).
    recording_dir : str
        Parent directory name of the video file (e.g. "250904_rEHT").
    recording : str
        Video recording name without extension.
    roi_name : str
        ROI name.
    frame_idx : int
        Frame index in the video.
    data_dir : str
        Directory to write to (e.g. ``<job-dir>/inferences``). Required.

    Returns
    -------
    dict
        The created sample entry.
    """
    _ensure_dir(data_dir)
    registry = _load_registry(data_dir)

    # Sample ID: {recording_dir}_{roi_name}_{recording}_{frame_no/30}
    sample_id = f"{recording_dir}_{roi_name}_{recording}_{frame_idx // 30:04d}"

    out_dir = _resolve_dir(data_dir)

    # Convert torch tensors to numpy if needed
    try:
        import torch
        if isinstance(roi_gray, torch.Tensor):
            roi_gray = roi_gray.detach().cpu().numpy()
    except ImportError:
        pass

    # Save ROI image (grayscale, uint8)
    roi_img = np.asarray(roi_gray, dtype=np.uint8)
    if roi_img.ndim > 2:
        roi_img = roi_img.squeeze()
    roi_filename = f"{sample_id}_ROI.jpg"
    roi_path = os.path.join(out_dir, roi_filename)
    if cv2 is not None:
        cv2.imwrite(roi_path, roi_img)
    else:
        from PIL import Image
        Image.fromarray(roi_img).save(roi_path)

    # Build coordinate list [[x1,y1], [x2,y2]]
    coords_list = coords.tolist() if isinstance(coords, np.ndarray) else list(coords)

    sample = {
        "id": sample_id,
        "roi_image": roi_filename,
        "heatmap_image": None,  # Generated during review
        "coords": coords_list,
        "model_name": model_name,
        "target_size": target_size,
        "recording_dir": recording_dir,
        "recording": recording,
        "roi_name": roi_name,
        "frame_idx": frame_idx,
        "checked": False,
    }

    # Replace existing sample with same ID, or append new
    existing_idx = None
    for i, s in enumerate(registry["samples"]):
        if s["id"] == sample_id:
            existing_idx = i
            break
    if existing_idx is not None:
        # Remove old ROI image if filename differs
        old_roi = registry["samples"][existing_idx].get("roi_image")
        if old_roi and old_roi != roi_filename:
            old_path = os.path.join(out_dir, old_roi)
            if os.path.isfile(old_path):
                os.remove(old_path)
        registry["samples"][existing_idx] = sample
    else:
        registry["samples"].append(sample)

    _save_registry(registry, data_dir)
    return sample


def _generate_heatmap(coords: list, roi_w: int, roi_h: int,
                      size: int = 256, sigma: float = 2.0) -> np.ndarray:
    """Generate a grayscale heatmap with Gaussian blobs at each coordinate.

    Parameters
    ----------
    coords : list of [x, y] in original ROI pixel space
    roi_w, roi_h : int
        Original ROI image dimensions for coordinate normalization.
    size : int
        Output heatmap dimension (square).
    sigma : float
        Gaussian sigma in output pixel units.

    Returns
    -------
    numpy array (size, size) uint8, black background with bright Gaussian blobs.
    """
    hm = np.zeros((size, size), dtype=np.float32)
    yg, xg = np.meshgrid(
        np.arange(size, dtype=np.float32),
        np.arange(size, dtype=np.float32),
        indexing="ij",
    )

    for cx, cy in coords:
        # Normalize to [0, 1] then scale to heatmap size
        px = (cx / roi_w) * size if roi_w > 0 else 0.0
        py = (cy / roi_h) * size if roi_h > 0 else 0.0
        blob = np.exp(-((xg - px) ** 2 + (yg - py) ** 2) / (2.0 * sigma ** 2))
        hm = np.maximum(hm, blob)

    hm_img = (np.clip(hm, 0.0, 1.0) * 255.0).astype(np.uint8)
    return hm_img


def update_coords_and_heatmap(sample_id: str, coords: list,
                              heatmap_size: int = 256,
                              blob_sigma: float = 2.0,
                              data_dir: Optional[str] = None) -> bool:
    """Update a sample's coordinates and generate its training heatmap.

    Called during review when the user approves a sample (ENTER).
    Loads the saved ROI image to determine original dimensions for
    coordinate normalization, renders a 256x256 Gaussian-blob heatmap,
    and updates the registry.

    Parameters
    ----------
    sample_id : str
    coords : list of [x, y] in original ROI pixel space
    heatmap_size : int
    blob_sigma : float
    data_dir : str
        Directory containing trainsets.json and images. Required.

    Returns
    -------
    bool — True on success
    """
    registry = _load_registry(data_dir)
    out_dir = _resolve_dir(data_dir)

    for s in registry["samples"]:
        if s["id"] == sample_id:
            # Load ROI image to get original dimensions
            roi_path = os.path.join(out_dir, s["roi_image"])
            roi = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE) if cv2 is not None else None
            if roi is None:
                return False
            roi_h, roi_w = roi.shape[:2]

            # Generate heatmap
            hm = _generate_heatmap(coords, roi_w, roi_h, heatmap_size, blob_sigma)
            hm_filename = f"{sample_id}_heatmap.jpg"
            hm_path = os.path.join(out_dir, hm_filename)
            if cv2 is not None:
                cv2.imwrite(hm_path, hm)
            else:
                from PIL import Image
                Image.fromarray(hm).save(hm_path)

            s["coords"] = coords
            s["heatmap_image"] = hm_filename
            _save_registry(registry, data_dir)
            return True
    return False


def sort_and_persist_registry(data_dir: Optional[str] = None) -> list[dict]:
    """Sort all samples by ID in-place and write back to JSON.

    Returns the sorted sample list.
    """
    registry = _load_registry(data_dir)
    registry["samples"].sort(key=lambda s: s["id"])
    _save_registry(registry, data_dir)
    return list(registry["samples"])


def deduplicate_samples(similarity_threshold: float = 0.05,
                        progress_callback=None,
                        data_dir: Optional[str] = None) -> int:
    """Remove duplicate adjacent samples based on ROI image similarity.

    Samples are sorted by ID first. Starting from the 2nd sample, each
    sample's ROI image is compared with the previous sample. If dimensions
    match, the per-pixel normalised absolute difference is computed. A pixel
    is considered "different" when its diff exceeds the threshold.
    If the proportion of different pixels is below the threshold, the later
    sample is considered a duplicate and deleted (JSON entry + image files).

    Parameters
    ----------
    similarity_threshold : float
        Both the per-pixel diff threshold and the max proportion of different
        pixels allowed before two samples are considered distinct.
    progress_callback : callable or None
        Called as progress_callback(current_pair_idx, total_pairs) before
        each comparison. Indices are 1-based for user display.
    data_dir : str
        Directory containing trainsets.json and images. Required.

    Returns the number of duplicates removed.
    """
    registry = _load_registry(data_dir)
    samples = registry["samples"]
    samples.sort(key=lambda s: s["id"])

    if len(samples) < 2:
        return 0

    out_dir = _resolve_dir(data_dir)
    removed = 0
    i = 1
    while i < len(samples):
        prev = samples[i - 1]
        curr = samples[i]

        if progress_callback is not None:
            progress_callback(i, len(samples) - 1)

        # Load ROI images
        prev_path = os.path.join(out_dir, prev["roi_image"])
        curr_path = os.path.join(out_dir, curr["roi_image"])

        prev_img = cv2.imread(prev_path, cv2.IMREAD_GRAYSCALE) if cv2 is not None else None
        curr_img = cv2.imread(curr_path, cv2.IMREAD_GRAYSCALE) if cv2 is not None else None

        if prev_img is None or curr_img is None:
            i += 1
            continue

        # Must have identical dimensions to compare
        if prev_img.shape != curr_img.shape:
            i += 1
            continue

        # Count pixels whose normalised difference exceeds the threshold
        diff = np.abs(prev_img.astype(np.float32) - curr_img.astype(np.float32)) / 256.0
        diff_pixel_count = int(np.sum(diff > similarity_threshold))
        diff_ratio = diff_pixel_count / diff.size

        if diff_ratio < similarity_threshold:
            # Delete duplicate image files
            for key in ("roi_image", "heatmap_image"):
                rel = curr.get(key)
                if rel:
                    abs_path = os.path.join(out_dir, rel)
                    if os.path.isfile(abs_path):
                        os.remove(abs_path)
            samples.pop(i)
            removed += 1
            # Don't increment i — next sample shifts into this position
        else:
            i += 1

    if removed > 0:
        _save_registry(registry, data_dir)

    return removed


def get_all_samples_sorted(data_dir: Optional[str] = None) -> list[dict]:
    """Return all samples sorted by sample ID (does not modify JSON)."""
    registry = _load_registry(data_dir)
    samples = list(registry["samples"])
    samples.sort(key=lambda s: s["id"])
    return samples


def get_unchecked_samples(data_dir: Optional[str] = None) -> list[dict]:
    """Return all samples with checked=False, sorted by sample ID."""
    return [s for s in get_all_samples_sorted(data_dir) if not s.get("checked", False)]


def mark_checked(sample_id: str, data_dir: Optional[str] = None) -> bool:
    """Mark a sample as checked (approved for training). Returns True on success."""
    registry = _load_registry(data_dir)
    for s in registry["samples"]:
        if s["id"] == sample_id:
            s["checked"] = True
            _save_registry(registry, data_dir)
            return True
    return False


def delete_sample(sample_id: str, data_dir: Optional[str] = None) -> bool:
    """Remove a sample and its image files. Returns True on success."""
    registry = _load_registry(data_dir)
    out_dir = _resolve_dir(data_dir)

    for i, s in enumerate(registry["samples"]):
        if s["id"] == sample_id:
            # Delete image files
            for key in ("roi_image", "heatmap_image"):
                rel = s.get(key)
                if rel:
                    abs_path = os.path.join(out_dir, rel)
                    if os.path.isfile(abs_path):
                        os.remove(abs_path)
            registry["samples"].pop(i)
            _save_registry(registry, data_dir)
            return True
    return False
