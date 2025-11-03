from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

# Lightweight, built-in file dialog (tkinter) for cross-platform CSV selection
try:
    import tkinter as tk
    from tkinter import filedialog as tk_filedialog
except Exception:  # pragma: no cover - environments without tkinter
    tk = None
    tk_filedialog = None

# Remember last selected CSV path for saving outputs nearby
_last_selected_csv_path: Optional[str] = None


def load_identified_csv() -> pd.DataFrame:
    """Open a file dialog to select an identified CSV and load required columns.

    Requirements:
    - The CSV must contain columns: 'time', 'force', and 'status' (case-insensitive accepted).
    - Returns a DataFrame containing exactly these columns in that order.

    Behavior:
    - If user cancels selection or file missing required columns, raises a descriptive error.
    - Coerces 'time' and 'force' to numeric; drops rows where either is NaN after coercion.
    - Preserves 'status' as-is (object/string).
    """
    file_path: Optional[str] = None
    if tk is not None and tk_filedialog is not None:
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            file_path = tk_filedialog.askopenfilename(
                title='Select identified CSV (time, force, status)',
                filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
            )
            root.destroy()
        except Exception:
            file_path = None

    if not file_path:
        raise RuntimeError('No CSV file selected.')

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    # cache for later saves
    global _last_selected_csv_path
    _last_selected_csv_path = file_path

    # Read CSV with a small encoding fallback
    try:
        df_try = pd.read_csv(file_path)
    except Exception:
        df_try = pd.read_csv(file_path, encoding='utf-8-sig')

    # Normalize column names for matching
    cols_original = list(df_try.columns)
    cols_lower = [str(c).strip().lower() for c in cols_original]
    required = {'time', 'force', 'status'}
    if not required.issubset(set(cols_lower)):
        raise ValueError(
            f"CSV must include columns {sorted(required)}; found: {cols_original}"
        )

    # Map lower names to their exact columns in the file
    col_map = {}
    for c in cols_original:
        cl = str(c).strip().lower()
        if cl in required and cl not in col_map:
            col_map[cl] = c

    # Build the canonical DataFrame with required columns in order
    out = pd.DataFrame({
        'time': pd.to_numeric(df_try[col_map['time']], errors='coerce'),
        'force': pd.to_numeric(df_try[col_map['force']], errors='coerce'),
        'status': df_try[col_map['status']].astype(object)
    })
    out = out.dropna(subset=['time', 'force']).reset_index(drop=True)
    return out


def seg_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """Segment a time-force-status DataFrame into contraction-peak-relaxation cycles.

    Returns a DataFrame with one row per cycle and columns:
      - time_arr: numpy array of time values in the cycle (object dtype)
      - force_arr: numpy array of force values in the cycle (object dtype)
      - force_start: force value at cycle start
      - force_peak: maximum force during the cycle
      - force_end: force value at cycle end
      - time_peak: time where force reaches the peak in this cycle (first occurrence)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['time_arr', 'force_arr', 'force_start', 'force_peak', 'force_end', 'time_peak', 't80_con', 't80_rel'])
    for col in ('time', 'force', 'status'):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is required in df for cycle segmentation.")

    # Work on a time-sorted copy for consistent segmentation
    sdf = df.sort_values('time', kind='mergesort').reset_index(drop=True)
    times = sdf['time'].to_numpy(dtype=float)
    forces = sdf['force'].to_numpy(dtype=float)
    statuses = sdf['status'].to_numpy(dtype=object)
    n = len(sdf)

    peak_idxs = np.flatnonzero(statuses == 'peak')
    if peak_idxs.size == 0:
        return pd.DataFrame(columns=['time_arr', 'force_arr', 'force_start', 'force_peak', 'force_end', 'time_peak', 't80_con', 't80_rel'])

    rows = []
    last_end = -1  # ensure non-overlapping cycles in sequence
    for p in peak_idxs:
        # Find left boundary: extend left over contiguous 'contraction'
        i = p - 1
        while i >= 0 and statuses[i] == 'contraction':
            i -= 1
        left = max(i + 1, 0)
        # Avoid overlapping with previous cycle end
        left = max(left, last_end + 1)

        # Find right boundary: extend right over contiguous 'relaxation'
        j = p + 1
        while j < n and statuses[j] == 'relaxation':
            j += 1
        right = min(j - 1, n - 1) if j > p + 1 else p  # if no relaxation, use the peak itself

        if right < left:
            # Degenerate; skip
            continue

        t_arr = times[left:right + 1]
        f_arr = forces[left:right + 1]
        if t_arr.size == 0:
            continue

        # Compute peak force and time within the window
        local_max_idx = int(np.argmax(f_arr))
        force_peak = float(f_arr[local_max_idx])
        time_peak = float(t_arr[local_max_idx])

        # Compute t80 metrics
        f_start = float(f_arr[0])
        f_end = float(f_arr[-1])
        # Time to 80% of peak rise (using 20% delta as specified)
        target_con = f_start + 0.2 * (force_peak - f_start)
        t80_con = np.nan
        # Find first index from start where force crosses target_con
        con_idx = np.argmax(f_arr >= target_con) if f_arr.size > 0 else 0
        if f_arr.size > 0 and (f_arr[con_idx] >= target_con):
            t80_con = float(t_arr[con_idx] - t_arr[0])

        # Relaxation: from peak to 20% above end (per spec)
        target_rel = f_end + 0.2 * (force_peak - f_end)
        t80_rel = np.nan
        if force_peak >= f_end:
            # typical decay, look for first <= target
            rel_mask = f_arr[local_max_idx:] <= target_rel
        else:
            # atypical increase, look for first >= target
            rel_mask = f_arr[local_max_idx:] >= target_rel
        if rel_mask.size > 0 and np.any(rel_mask):
            rel_offset = int(np.argmax(rel_mask))
            rel_idx = local_max_idx + rel_offset
            t80_rel = float(t_arr[rel_idx] - t_arr[local_max_idx])

        row = {
            'time_arr': t_arr.copy(),
            'force_arr': f_arr.copy(),
            'force_start': f_start,
            'force_peak': force_peak,
            'force_end': f_end,
            'time_peak': time_peak,
            't80_con': t80_con,
            't80_rel': t80_rel,
        }
        rows.append(row)
        last_end = right

    cycles_df = pd.DataFrame(rows, columns=['time_arr', 'force_arr', 'force_start', 'force_peak', 'force_end', 'time_peak', 't80_con', 't80_rel'])
    return cycles_df


def analyze_data(lt_identified_df: pd.DataFrame, cycles_df: pd.DataFrame) -> pd.DataFrame:
    """Compute high-level metrics from identified data and cycles.

    Returns a single-row DataFrame with columns:
      'EHT name', 'Day', 'Contraction Force', 'Relaxation Force', 'Diastolic Tension',
      'Frequency', 'Time to Peak 80%', 'Relaxation Time 80%'
    """
    # Contraction force: mean peak force across cycles
    force_con = float(np.nanmean(cycles_df['force_peak'])) if not cycles_df.empty else np.nan

    # Relaxation force: mean of forces at 'rest' timepoints
    if lt_identified_df is not None and not lt_identified_df.empty:
        rest_forces = pd.to_numeric(lt_identified_df.loc[lt_identified_df['status'] == 'rest', 'force'], errors='coerce')
        force_rest = float(np.nanmean(rest_forces)) if rest_forces.size > 0 else np.nan
    else:
        force_rest = np.nan

    # Diastolic/burst tension baseline: mean of cycle start/end forces and 'burst rest' forces combined
    parts = []
    if not cycles_df.empty:
        parts.append(pd.to_numeric(cycles_df['force_start'], errors='coerce').to_numpy())
        parts.append(pd.to_numeric(cycles_df['force_end'], errors='coerce').to_numpy())
    if lt_identified_df is not None and not lt_identified_df.empty:
        burst_forces = pd.to_numeric(lt_identified_df.loc[lt_identified_df['status'] == 'burst rest', 'force'], errors='coerce').to_numpy()
        if burst_forces.size > 0:
            parts.append(burst_forces)
    if len(parts) > 0:
        force_burst_rest = float(np.nanmean(np.concatenate(parts)))
    else:
        force_burst_rest = np.nan

    # Frequency: beats per minute (BPM) = peaks per second * 60
    if lt_identified_df is not None and not lt_identified_df.empty:
        t_min = float(np.nanmin(lt_identified_df['time']))
        t_max = float(np.nanmax(lt_identified_df['time']))
        total_time = t_max - t_min  # assume 'time' is in seconds
        n_peaks = int(np.sum(lt_identified_df['status'] == 'peak'))
        freq = ((n_peaks / total_time) * 60.0) if total_time > 0 else np.nan
    else:
        freq = np.nan

    # t80 metrics means
    t80_con_mean = float(np.nanmean(cycles_df['t80_con'])) if ('t80_con' in cycles_df.columns and not cycles_df.empty) else np.nan
    t80_rel_mean = float(np.nanmean(cycles_df['t80_rel'])) if ('t80_rel' in cycles_df.columns and not cycles_df.empty) else np.nan

    # Diastolic tension
    dt = force_burst_rest - force_rest if (np.isfinite(force_burst_rest) and np.isfinite(force_rest)) else np.nan

    analysis_df = pd.DataFrame([
        {
            'EHT name': '',
            'Day': '',
            'Contraction Force': force_con,
            'Relaxation Force': force_rest,
            'Diastolic Tension': dt,
            'Frequency': freq,
            'Time to Peak 80%': t80_con_mean,
            'Relaxation Time 80%': t80_rel_mean,
        }
    ])
    return analysis_df


if __name__ == '__main__':
    df = load_identified_csv()
    print(f"Loaded identified DataFrame: {df.shape[0]} rows, {df.shape[1]} cols")
    
    # Segment into contraction-peak-relaxation cycles
    cycles_df = seg_cycles(df)
    print(f"Segmented into {len(cycles_df)} cycles")
    with pd.option_context('display.max_rows', 10, 'display.width', 160):
        print(cycles_df.head(5))

    # Analyze and save
    analysis_df = analyze_data(df, cycles_df)
    print("Analysis summary:")
    print(analysis_df)

    # Save next to the selected CSV
    out_path = None
    if _last_selected_csv_path and os.path.isfile(_last_selected_csv_path):
        base_dir = os.path.dirname(_last_selected_csv_path)
        base_name = os.path.splitext(os.path.basename(_last_selected_csv_path))[0]
        out_path = os.path.join(base_dir, f"{base_name}_analysis.csv")
    else:
        out_path = os.path.join(os.getcwd(), 'analysis.csv')

    analysis_df.to_csv(out_path, index=False)
    print(f"Saved analysis to: {out_path}")

