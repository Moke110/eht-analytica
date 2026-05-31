"""Analysis service — wraps LengthDataAnalyzer for the API."""

from __future__ import annotations

import os
import threading

import pandas as pd

from functions.length_data_analyzer import LengthDataAnalyzer
from backend.services.task_manager import task_manager


def validate_paths(paths: list[str]) -> tuple[list[str], list[str]]:
    valid, invalid = [], []
    for p in paths:
        if os.path.isfile(p) and p.lower().endswith('.csv'):
            valid.append(p)
        else:
            invalid.append(p)
    return valid, invalid


def start_analysis(csv_paths: list[str]) -> str:
    """Run analysis pipeline in background. Returns task_id."""
    tid = task_manager.create("analysis")

    def worker():
        try:
            analyzer = LengthDataAnalyzer()
            total = len(csv_paths)
            all_fs: list[pd.DataFrame] = []
            all_metrics: list[pd.DataFrame] = []

            for i, path in enumerate(csv_paths):
                if task_manager.is_cancelled(tid):
                    raise InterruptedError("Task cancelled")

                task_manager.update(tid, (i / total) * 100, f"Analyzing {i+1}/{total}: {os.path.basename(path)}")
                fs_dfs, metrics_df = analyzer.analyze_lt_csvs([path])
                all_fs.extend(fs_dfs)
                all_metrics.append(metrics_df)

            # Serialize results
            fs_serialized = []
            for df in all_fs:
                fs_serialized.append({
                    "eht_name": df.attrs.get("EHT_name", ""),
                    "data": df.to_dict(orient="records"),
                })

            metrics_combined = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
            metrics_rows = metrics_combined.to_dict(orient="records") if not metrics_combined.empty else []

            task_manager.complete(tid, {
                "fs_results": fs_serialized,
                "metrics_rows": metrics_rows,
                "files_processed": total,
            })
        except InterruptedError:
            pass
        except Exception as e:
            task_manager.fail(tid, str(e))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return tid


def save_results(fs_results: list[dict], metrics_rows: list[dict] | None, output_folder: str) -> dict:
    """Write analysis results to CSV files."""
    os.makedirs(output_folder, exist_ok=True)
    saved = []

    for item in fs_results:
        name = item.get("eht_name", "unknown")
        data = item.get("data", [])
        if data:
            df = pd.DataFrame(data)
            csv_path = os.path.join(output_folder, f"{name}_analyzed.csv")
            df.to_csv(csv_path, index=False)
            saved.append(os.path.basename(csv_path))

    metrics_path = None
    if metrics_rows:
        df = pd.DataFrame(metrics_rows)
        metrics_path = os.path.join(output_folder, "EHT_metrics.csv")
        df.to_csv(metrics_path, index=False)
        saved.append(os.path.basename(metrics_path))

    return {"saved_files": saved, "metrics_path": metrics_path}
