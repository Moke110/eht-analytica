"""Per-recording job configuration — manages <video-dir>/EHT-analytics/config.json."""

from __future__ import annotations

import json
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

JOB_DIR_NAME = "EHT-analytics"
CONFIG_FILE = "config.json"


def _job_dir(video_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(video_path)), JOB_DIR_NAME)


def _config_path(job_dir: str) -> str:
    return os.path.join(job_dir, CONFIG_FILE)


def _read(job_dir: str) -> dict:
    cp = _config_path(job_dir)
    if not os.path.isfile(cp):
        return {"samples": []}
    try:
        with open(cp, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Migrate old key name
        if "accessions" in data:
            data["samples"] = data.pop("accessions")
        return data
    except json.JSONDecodeError:
        logger.exception("Corrupted config.json in %s — returning empty fallback", job_dir)
        return {"samples": [], "metadata_keys": []}
    except Exception:
        logger.exception("Failed to read config.json in %s", job_dir)
        return {"samples": [], "metadata_keys": []}


def _write(job_dir: str, data: dict) -> None:
    os.makedirs(job_dir, exist_ok=True)
    cp = _config_path(job_dir)
    tmp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                      dir=job_dir, delete=False,
                                      suffix=".tmp", prefix="config_")
    try:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp.close()
        os.replace(tmp.name, cp)
    except Exception:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise


def init_job_dir(video_path: str) -> str:
    """Create and return the job directory path for a video."""
    jd = _job_dir(video_path)
    os.makedirs(jd, exist_ok=True)
    # Ensure config.json exists
    cp = _config_path(jd)
    if not os.path.isfile(cp):
        _write(jd, {"samples": []})
    return jd


def add_samples(job_dir: str, video_path: str, roi_names: list[str],
                   length_csv_prefix: str = "") -> list[dict]:
    """Add samples for each ROI of a video. Skips existing (recording, roi) pairs. Returns entries."""
    data = _read(job_dir)
    existing = data.get("samples", [])

    # Check for existing entries with same recording+roi — skip duplicates
    recording_name = os.path.splitext(os.path.basename(video_path))[0]
    seen = {(a.get("recording_name"), a.get("roi_name")) for a in existing}
    new_roi_names = [r for r in roi_names if (recording_name, r) not in seen]

    if not new_roi_names:
        # All ROIs already exist — return existing matching entries
        return [a for a in existing
                if a.get("recording_name") == recording_name and a.get("roi_name") in roi_names]

    # Determine next ID
    max_id = -1
    for a in existing:
        try:
            max_id = max(max_id, int(a.get("id", "-1")))
        except (ValueError, TypeError):
            pass
    next_id = max_id + 1

    new_entries = []
    for roi_name in new_roi_names:
        entry = {
            "id": f"{next_id:05d}",
            "recording_name": recording_name,
            "roi_name": roi_name,
            "length_csv": "",
            "force_status_csv": "",
            "metrics_csv": "",
            "metadata": {},
        }
        existing.append(entry)
        new_entries.append(entry)
        next_id += 1

    data["samples"] = existing
    _write(job_dir, data)
    return new_entries


def update_sample_field(job_dir: str, sample_id: str, field: str,
                           rel_path: str) -> None:
    """Update a specific field of a sample with a relative path."""
    data = _read(job_dir)
    found = False
    for a in data.get("samples", []):
        if a.get("id") == sample_id:
            a[field] = rel_path
            found = True
            break
    if not found:
        logger.warning("update_sample_field: sample_id %s not found in %s (field=%s)",
                       sample_id, job_dir, field)
    _write(job_dir, data)


def get_job_config(job_dir: str) -> dict:
    """Return the full job config."""
    data = _read(job_dir)
    return {
        "samples": data.get("samples", []),
        "metadata_keys": data.get("metadata_keys", []),
    }


def save_metadata(job_dir: str, metadata_keys: list[str],
                  sample_updates: list[dict]) -> None:
    """Save metadata keys and per-accession metadata values."""
    data = _read(job_dir)
    data["metadata_keys"] = metadata_keys
    samples = data.get("samples", [])
    update_map = {a["id"]: a.get("metadata", {}) for a in sample_updates}
    for acc in samples:
        if acc["id"] in update_map:
            acc["metadata"] = update_map[acc["id"]]
    _write(job_dir, data)


def resolve_path(job_dir: str, rel_path: str) -> str:
    """Resolve a relative path from config to an absolute path."""
    if not rel_path:
        return ""
    return os.path.normpath(os.path.join(job_dir, rel_path))
