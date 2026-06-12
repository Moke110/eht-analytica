"""Report plot data computation — group metrics by user-selected dimensions."""

from __future__ import annotations

import csv
import logging
import math
import os

from backend.services.job_config import get_job_config, resolve_path

logger = logging.getLogger(__name__)


def _resolve_group_label(sample: dict, group_keys: list[str]) -> str:
    parts = []
    for key in group_keys:
        if key == "Recording":
            parts.append(sample.get("recording_name", ""))
        elif key == "ROI Name":
            parts.append(sample.get("roi_name", ""))
        else:
            parts.append(sample.get("metadata", {}).get(key, ""))
    return "_".join(parts)


def compute_plot_data(job_dir: str, sample_ids: list[str],
                      group_keys: list[str]) -> dict:
    """Compute per-group statistics for each metric across selected samples.

    Returns a dict matching ReportsPlotResponse schema.
    """
    if not job_dir or not sample_ids or not group_keys:
        return {"metrics": [], "groups": [], "data": {}}

    config = get_job_config(job_dir)
    samples_by_id: dict[str, dict] = {
        s["id"]: s for s in config.get("samples", [])
    }

    # Collect raw values: {(group_label, metric_name): [values]}
    raw: dict[tuple[str, str], list[float]] = {}
    group_order: list[str] = []

    for sid in sample_ids:
        sample = samples_by_id.get(sid)
        if not sample:
            continue

        metrics_csv = sample.get("metrics_csv", "")
        if not metrics_csv:
            continue

        csv_path = resolve_path(job_dir, metrics_csv)
        if not os.path.isfile(csv_path):
            logger.warning("Metrics CSV not found: %s", csv_path)
            continue

        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if not rows:
                    continue
                row = rows[0]
        except Exception:
            logger.warning("Failed to read metrics CSV: %s", csv_path)
            continue

        group_label = _resolve_group_label(sample, group_keys)
        if group_label not in group_order:
            group_order.append(group_label)

        for col, val in row.items():
            if col == "EHT name":
                continue
            try:
                num = float(val)
            except (ValueError, TypeError):
                continue
            key = (group_label, col)
            raw.setdefault(key, []).append(num)

    # Collect all metric names
    metric_names = sorted({m for (_, m) in raw})
    groups = group_order

    # Compute statistics
    data: dict[str, dict[str, dict]] = {}
    for metric in metric_names:
        data[metric] = {}
        for group in groups:
            vals = raw.get((group, metric), [])
            n = len(vals)
            if n == 0:
                data[metric][group] = {
                    "sample_values": [],
                    "mean": 0.0,
                    "ste": 0.0,
                }
                continue
            mean = sum(vals) / n
            if n > 1:
                variance = sum((v - mean) ** 2 for v in vals) / (n - 1)
                std = math.sqrt(variance)
                ste = std / math.sqrt(n)
            else:
                ste = 0.0
            data[metric][group] = {
                "sample_values": vals,
                "mean": round(mean, 6),
                "ste": round(ste, 6),
            }

    return {
        "metrics": metric_names,
        "groups": groups,
        "data": data,
    }


def compute_full_data(job_dir: str, sample_ids: list[str]) -> dict:
    """Return per-sample data with metadata and metrics values for CSV export.

    Returns a dict with keys: samples, metrics, metadata_keys.
    """
    if not job_dir or not sample_ids:
        return {"samples": [], "metrics": [], "metadata_keys": []}

    config = get_job_config(job_dir)
    samples_by_id: dict[str, dict] = {
        s["id"]: s for s in config.get("samples", [])
    }
    all_metadata_keys = list(config.get("metadata_keys", []))

    result_samples: list[dict] = []
    all_metrics: set[str] = set()

    for sid in sample_ids:
        sample = samples_by_id.get(sid)
        if not sample:
            continue

        metrics = {}
        metrics_csv = sample.get("metrics_csv", "")
        if metrics_csv:
            csv_path = resolve_path(job_dir, metrics_csv)
            if os.path.isfile(csv_path):
                try:
                    with open(csv_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        if rows:
                            row = rows[0]
                            for col, val in row.items():
                                if col == "EHT name":
                                    continue
                                try:
                                    num = float(val)
                                except (ValueError, TypeError):
                                    num = val
                                metrics[col] = num
                                all_metrics.add(col)
                except Exception:
                    logger.warning("Failed to read metrics CSV: %s", csv_path)

        result_samples.append({
            "id": sample.get("id", ""),
            "recording_name": sample.get("recording_name", ""),
            "roi_name": sample.get("roi_name", ""),
            "metadata": sample.get("metadata", {}),
            "metrics": metrics,
        })

    return {
        "samples": result_samples,
        "metrics": sorted(all_metrics),
        "metadata_keys": all_metadata_keys,
    }
