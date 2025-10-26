"""
Time-length preprocessing utilities.

Functions:
- load(): Open a dialog to select a CSV and load into a pandas DataFrame.
- t_l_pre_process(df): Make time values regular at the minimal interval and linearly fill gaps.
- plot_t_l(df): Plot as a blue line with adaptive figure width so neighboring time spots have >=3px spacing.

Run this file directly to choose a CSV, preprocess it, and plot.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _infer_time_length_columns(df: pd.DataFrame) -> Tuple[str, str]:
	"""Infer the time and length column names.

	Preference order:
	1) Columns containing 'time' and 'length' (case-insensitive)
	2) First two numeric columns

	Raises:
		ValueError: If suitable columns cannot be determined.
	"""

	cols_lower = {c.lower(): c for c in df.columns}

	# Try name-based inference
	time_col = None
	length_col = None
	for key in cols_lower:
		if "time" in key and time_col is None:
			time_col = cols_lower[key]
		if ("length" in key or key in {"len", "value", "val", "y"}) and length_col is None:
			length_col = cols_lower[key]

	# Avoid picking the same column twice if names are ambiguous
	if time_col == length_col:
		length_col = None

	# Fallback to first two numeric columns
	if time_col is None or length_col is None:
		numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
		if len(numeric_cols) >= 2:
			if time_col is None:
				time_col = numeric_cols[0]
			if length_col is None:
				# choose the first numeric column different from time
				for c in numeric_cols:
					if c != time_col:
						length_col = c
						break

	if not time_col or not length_col:
		raise ValueError(
			"Could not infer time and length columns. Ensure your CSV has columns named like 'time' and 'length' or at least two numeric columns."
		)

	return time_col, length_col


def load() -> Optional[tuple[pd.DataFrame, str]]:
	"""Open a file dialog to select a .csv containing time-length data and load it.

	Returns None if the selection is canceled.
	"""

	try:
		# Lazy import to avoid tkinter import cost/issues when unused
		import tkinter as tk
		from tkinter import filedialog

		root = tk.Tk()
		root.withdraw()  # Hide the main window
		file_path = filedialog.askopenfilename(
			title="Select time-length CSV",
			filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
		)
		root.destroy()
	except Exception as e:  # e.g., TclError in headless
		raise RuntimeError(f"Unable to open file dialog: {e}")

	if not file_path:
		return None

	if not os.path.isfile(file_path):
		raise FileNotFoundError(f"File not found: {file_path}")

	# Read CSV; let pandas infer the delimiter
	df = pd.read_csv(file_path)

	if df.empty:
		raise ValueError("Loaded CSV is empty.")

	# Infer columns and standardize to ['time', 'length']
	t_col, l_col = _infer_time_length_columns(df)
	df = df[[t_col, l_col]].rename(columns={t_col: "time", l_col: "length"})

	# Coerce to numeric and clean
	df["time"] = pd.to_numeric(df["time"], errors="coerce")
	df["length"] = pd.to_numeric(df["length"], errors="coerce")
	df = df.dropna(subset=["time", "length"]).sort_values("time")
	df = df.drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)

	if len(df) < 1:
		raise ValueError("No valid numeric 'time' and 'length' rows after cleaning.")

	return df, file_path


def t_l_pre_process(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[float]]:
	"""Preprocess a time-length DataFrame to a regular time grid.

	Steps:
	- Determine minimal positive time gap as the standard interval.
	- Create a regular time grid at n * standard_interval from 0 to the last time.
	- Fill values linearly between known points; no extrapolation beyond the min/max known time.

	Returns:
		(processed_df, standard_interval)
	"""

	if df is None or len(df) == 0:
		return df, None

	df = df.copy()
	if "time" not in df.columns or "length" not in df.columns:
		raise ValueError("DataFrame must contain 'time' and 'length' columns.")

	df = df.sort_values("time").reset_index(drop=True)

	# Compute minimal positive time gap
	t_values = df["time"].to_numpy(dtype=float)
	diffs = np.diff(np.unique(t_values))
	positive_diffs = diffs[diffs > 0]

	if positive_diffs.size == 0:
		# Single time point or all times equal: nothing to resample
		return df.copy(), None

	standard_interval = float(np.min(positive_diffs))

	# Build regular grid from 0 to last time (inclusive)
	t_last = float(np.max(t_values))
	if standard_interval <= 0:
		# Safety; shouldn't occur due to positive_diffs check
		return df.copy(), None

	# Use a small epsilon to ensure inclusion of the endpoint due to FP rounding
	eps = standard_interval * 1e-9
	grid = np.arange(0.0, t_last + eps + standard_interval, standard_interval)

	# Interpolate within the range [t_min, t_max]; don't extrapolate
	t_min = float(np.min(t_values))
	t_max = t_last
	y_values = df["length"].to_numpy(dtype=float)

	# For interpolation, use numpy.interp which clips outside to edge values; we will mask them after.
	interp_y = np.interp(grid, t_values, y_values)

	# Mask out-of-range grid points strictly before first or after last known time to avoid extrapolation
	mask_in_range = (grid >= t_min - eps) & (grid <= t_max + eps)
	grid_in = grid[mask_in_range]
	y_in = interp_y[mask_in_range]

	out = pd.DataFrame({"time": grid_in, "length": y_in})

	# Ensure exact multiples (reduce FP noise) by rounding to a sensible number of decimals
	# Determine decimals from interval magnitude
	if standard_interval < 1:
		# e.g., 0.01 -> 2 decimals; add a buffer
		decimals = min(9, max(0, int(abs(np.floor(np.log10(standard_interval)))) + 2))
	else:
		decimals = 6
	out["time"] = out["time"].round(decimals)

	return out.reset_index(drop=True), standard_interval


def detect_movement_1(df: pd.DataFrame, *, interval: Optional[float] = None) -> pd.DataFrame:
	"""Classify movement per row into one of four states:
	- baseline: slight vibration around baseline (small slope magnitude)
	- contraction: rapidly decreasing length (negative slope beyond threshold)
	- relaxation: rapidly increasing length (positive slope beyond threshold)
	- peak: local minimum where slope flips from negative to non-negative

	Heuristic based on first derivative with a robust threshold from MAD.

	Args:
		df: DataFrame with columns ['time', 'length'] (regular grid preferred).
		interval: Optional known time step. If None, computed from data.

	Returns:
		A copy of df with a new 'movement' column containing one of
		{'baseline','contraction','relaxation','peak'} for each row.
	"""

	if df is None or df.empty:
		return df

	dfx = df.copy()
	if "time" not in dfx.columns or "length" not in dfx.columns:
		raise ValueError("DataFrame must contain 'time' and 'length' columns.")

	t = dfx["time"].to_numpy(dtype=float)
	y = dfx["length"].to_numpy(dtype=float)
	if len(t) < 3:
		# Too short for derivative-based classification
		dfx["movement"] = "baseline"
		return dfx

	# Determine spacing (use median diff to be robust)
	if interval is None:
		diffs = np.diff(t)
		positive = diffs[diffs > 0]
		interval = float(np.median(positive)) if positive.size else 1.0

	# First derivative (dy/dt)
	dy_dt = np.gradient(y, t)

	# Robust threshold using MAD
	med = float(np.median(dy_dt))
	mad = float(np.median(np.abs(dy_dt - med)))
	# Convert MAD to approx std for normal dist
	robust_sigma = 1.4826 * mad
	thr = max(3.0 * robust_sigma, 1e-12)

	labels = np.full(len(y), "baseline", dtype=object)
	labels[dy_dt <= -thr] = "contraction"
	labels[dy_dt >= thr] = "relaxation"

	# Peaks: local minima with slope sign change (- -> +)
	for i in range(1, len(y) - 1):
		if dy_dt[i - 1] < 0.0 and dy_dt[i] >= 0.0 and y[i] <= y[i - 1] and y[i] <= y[i + 1]:
			labels[i] = "peak"

	dfx["movement"] = pd.Categorical(labels, categories=["baseline", "contraction", "relaxation", "peak"], ordered=False)
	return dfx


def plot_t_l(
	df: pd.DataFrame,
	title: Optional[str] = None,
	*,
	csv_path: Optional[str] = None,
	x_end_time: Optional[float] = None,
) -> None:
	"""Plot time-length data as a blue line with adaptive width.

	Ensures at least 3 pixels between neighboring time spots by adjusting figure width.
	"""

	if df is None or df.empty:
		raise ValueError("Nothing to plot: DataFrame is empty.")

	n = len(df)
	# Compute required width in pixels: ensure >= 3px per gap
	gaps = max(n - 1, 1)
	min_gap_px = 1
	axis_padding_px = 50  # space for y-axis labels/margins
	min_width_px = 640
	required_px = max(min_width_px, gaps * min_gap_px + axis_padding_px)

	dpi = 100
	width_in = required_px / dpi
	height_in = 4.0  # reasonable default height

	plt.figure(figsize=(width_in, height_in), dpi=dpi)
	ax = plt.gca()

	# Shade background per movement between midpoints of neighboring time points
	# If 'movement' not present, fall back to baseline shading
	move_series = df["movement"].astype(str) if "movement" in df.columns else pd.Series(["baseline"] * len(df))
	t = df["time"].to_numpy(dtype=float)
	# Determine segment bounds
	lefts = np.empty_like(t)
	rights = np.empty_like(t)
	lefts[0] = max(0.0, t[0] - (t[1] - t[0]) / 2.0) if len(t) > 1 else 0.0
	rights[-1] = float(x_end_time) if x_end_time is not None else (t[-1] + (t[-1] - t[-2]) / 2.0 if len(t) > 1 else t[-1])
	for i in range(1, len(t)):
		lefts[i] = (t[i - 1] + t[i]) / 2.0
	for i in range(0, len(t) - 1):
		rights[i] = (t[i] + t[i + 1]) / 2.0
	if x_end_time is not None:
		lefts = np.clip(lefts, 0.0, float(x_end_time))
		rights = np.clip(rights, 0.0, float(x_end_time))

	colors = {
		"baseline": (0.80, 0.89, 1.00, 0.35),  # light blue
		"contraction": (1.00, 0.95, 0.70, 0.35),  # light yellow
		"peak": (1.00, 0.80, 0.80, 0.40),  # light red
		"relaxation": (0.80, 1.00, 0.80, 0.35),  # light green
	}
	for i in range(len(t)):
		c = colors.get(move_series.iloc[i], (0.95, 0.95, 0.95, 0.2))
		l = float(lefts[i])
		r = float(rights[i])
		if r > l:
			ax.axvspan(l, r, facecolor=c, edgecolor=None, linewidth=0, zorder=0)

	# Plot line on top
	plt.plot(df["time"], df["length"], color="blue", linewidth=1.5, zorder=5)
	plt.xlabel("Time")
	plt.ylabel("Length")
	# X-axis from 0 to last given time point
	if x_end_time is None:
		x_end_time = float(np.max(df["time"].to_numpy()))
	plt.xlim(0.0, float(x_end_time))
	if title:
		plt.title(title)
	plt.grid(True, linestyle=":", alpha=0.3)
	plt.tight_layout()
	# Save to same directory as the given CSV (if provided)
	if csv_path:
		try:
			base = os.path.splitext(os.path.basename(csv_path))[0]
			out_name = f"{base}_t_l_plot.png"
			out_dir = os.path.dirname(csv_path) or os.getcwd()
			out_path = os.path.join(out_dir, out_name)
			plt.savefig(out_path)
			print(f"Saved plot to: {out_path}")
		except Exception as e:
			print(f"Warning: failed to save plot next to CSV: {e}")
	plt.show()


if __name__ == "__main__":
	# 1) Load CSV via dialog
	loaded = load()
	if loaded is None:
		print("Selection canceled. Exiting.")
	else:
		data, csv_path = loaded
		# 2) Preprocess to regular grid with linear filling
		processed, std_interval = t_l_pre_process(data)
		# 2.5) Detect movement per row
		processed = detect_movement_1(processed, interval=std_interval)
		if std_interval is not None:
			ttl = f"Time-Length (interval={std_interval:g})"
		else:
			ttl = "Time-Length"

		# 3) Plot (x-axis 0..last given time; save next to CSV before showing)
		x_end = float(data["time"].max())
		plot_t_l(processed, title=ttl, csv_path=csv_path, x_end_time=x_end)

