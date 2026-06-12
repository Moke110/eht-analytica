"""Analysis service — wraps LengthDataAnalyzer for the API."""

from __future__ import annotations

import os
import threading
import traceback

import pandas as pd

from functions.length_data_analyzer import LengthDataAnalyzer
from backend.services.task_manager import task_manager
from backend.services.job_config import get_job_config, update_sample_field


def validate_paths(paths: list[str]) -> tuple[list[str], list[str]]:
    valid, invalid = [], []
    for p in paths:
        if os.path.isfile(p) and p.lower().endswith('.csv'):
            valid.append(p)
        else:
            invalid.append(p)
    return valid, invalid


def _check_csv_readable(path: str) -> None:
    """Raise ValueError if the CSV header prevents use as time/length input."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        df = pd.read_csv(path, nrows=0)
    except FileNotFoundError:
        raise
    except Exception:
        raise ValueError(f"Unreadable CSV: {os.path.basename(path)}")

    cols_lower = [str(c).strip().lower() for c in df.columns]
    if 'time' not in cols_lower or 'length' not in cols_lower:
        raise ValueError(
            f"CSV must contain 'time' and 'length' columns. "
            f"Found columns: {list(df.columns)} in {os.path.basename(path)}"
        )


def start_analysis(csv_paths: list[str], job_dir: str | None = None) -> str:
    """Run analysis pipeline in background. Returns task_id.

    Raises ValueError synchronously if any CSV is invalid.
    """
    # Pre-flight validation — fails fast before spawning worker thread
    for path in csv_paths:
        _check_csv_readable(path)

    tid = task_manager.create("analysis")

    def worker():
        output_dir = job_dir or os.path.dirname(csv_paths[0])
        try:
            analyzer = LengthDataAnalyzer()
            total = len(csv_paths)
            all_fs: list[pd.DataFrame] = []
            all_metrics: list[pd.DataFrame] = []

            for i, path in enumerate(csv_paths):
                if task_manager.is_cancelled(tid):
                    return

                task_manager.update(tid, (i / total) * 100, f"Analyzing {i+1}/{total}: {os.path.basename(path)}")
                fs_dfs, metrics_df = analyzer.analyze_lt_csvs([path])
                all_fs.extend(fs_dfs)
                all_metrics.append(metrics_df)

                if task_manager.is_cancelled(tid):
                    return

                task_manager.update(tid, ((i + 1) / total) * 100, f"Done {i+1}/{total}")

            fs_serialized = []
            for df in all_fs:
                fs_serialized.append({
                    "eht_name": df.attrs.get("EHT_name", ""),
                    "data": df.to_dict(orient="records"),
                })

            metrics_combined = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
            metrics_rows = metrics_combined.to_dict(orient="records") if not metrics_combined.empty else []

            # Save results to disk immediately — not dependent on frontend callback
            saved = save_results(fs_serialized, metrics_rows, output_dir, job_dir)

            task_manager.complete(tid, {
                "fs_results": fs_serialized,
                "metrics_rows": metrics_rows,
                "files_processed": total,
                "saved_files": saved.get("saved_files", []),
                "metrics_path": saved.get("metrics_path"),
            })
        except Exception as e:
            traceback.print_exc()
            task_manager.fail(tid, str(e))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return tid


def save_results(fs_results: list[dict], metrics_rows: list[dict] | None,
                 output_folder: str, job_dir: str | None = None) -> dict:
    """Write analysis results to force-status/ and metrics/ subfolders."""
    os.makedirs(output_folder, exist_ok=True)
    fs_dir = os.path.join(output_folder, "force-status")
    metrics_dir = os.path.join(output_folder, "metrics")
    os.makedirs(fs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    saved = []

    # Build a lookup dict from eht_name -> accession once
    eht_to_acc: dict[str, dict] = {}
    if job_dir:
        config = get_job_config(job_dir)
        for acc in config.get("samples", []):
            lc = acc.get("length_csv", "")
            if lc:
                # lc is like "lengths/00000_d18.avi_EHT-1_length.csv"
                # Strip prefix and suffix to derive the eht_name
                base = os.path.basename(lc)
                if base.endswith("_length.csv"):
                    eht_name = base[:-len("_length.csv")]
                    eht_to_acc[eht_name] = acc

    for item in fs_results:
        name = item.get("eht_name", "unknown")
        data = item.get("data", [])
        if not data:
            continue

        acc = eht_to_acc.get(name) if job_dir else None
        if acc:
            filename = f"{acc['id']}_{acc['recording_name']}_{acc['roi_name']}_force_status.csv"
        else:
            filename = f"{name}_force_status.csv"

        df = pd.DataFrame(data)
        csv_path = os.path.join(fs_dir, filename)
        df.to_csv(csv_path, index=False)
        rel = f"force-status/{filename}"
        saved.append(rel)

        if job_dir and acc:
            update_sample_field(job_dir, acc["id"], "force_status_csv", rel)
            acc["force_status_csv"] = rel  # update in-memory copy for metrics loop below

    metrics_path = None
    if metrics_rows and job_dir:
        for row in metrics_rows:
            eht_name = row.get("EHT name", "")
            acc = eht_to_acc.get(eht_name) if eht_name else None
            if acc and acc.get("force_status_csv"):
                filename = f"{acc['id']}_{acc['recording_name']}_{acc['roi_name']}_metrics.csv"
                single = pd.DataFrame([row])
                csv_path = os.path.join(metrics_dir, filename)
                single.to_csv(csv_path, index=False)
                rel = f"metrics/{filename}"
                saved.append(rel)
                update_sample_field(job_dir, acc["id"], "metrics_csv", rel)
        metrics_path = metrics_dir
    elif metrics_rows:
        df = pd.DataFrame(metrics_rows)
        filename = "EHT_metrics.csv"
        csv_path = os.path.join(metrics_dir, filename)
        df.to_csv(csv_path, index=False)
        rel = f"metrics/{filename}"
        saved.append(rel)
        metrics_path = metrics_dir

    return {"saved_files": saved, "metrics_path": metrics_path}


def _find_sample_for_eht(job_dir: str, eht_name: str) -> dict | None:
    """Find the accession whose length_csv ends with {eht_name}_length.csv."""
    if not job_dir or not eht_name:
        return None
    config = get_job_config(job_dir)
    for acc in config.get("samples", []):
        lc = acc.get("length_csv", "")
        if lc.endswith(f"{eht_name}_length.csv"):
            return acc
    return None
