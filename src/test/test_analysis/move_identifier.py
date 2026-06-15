from __future__ import annotations

import os
import math
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde

# Use a lightweight, built-in file dialog for CSV selection on all platforms
try:
	# tkinter is part of the stdlib; this import may fail only on very minimal Python builds
	import tkinter as tk
	from tkinter import filedialog as tk_filedialog
except Exception:  # pragma: no cover - fallback when tkinter is unavailable
	tk = None
	tk_filedialog = None


# Module-level storage of the last selected CSV path so we can save plots next to it
_last_selected_csv_path: str | None = None


def _ensure_time_length_df(df: pd.DataFrame) -> pd.DataFrame:
	"""Coerce an arbitrary two-column CSV into a DataFrame with ['time', 'length'].

	- If headers contain 'time' and 'length', use them.
	- If there are exactly two columns with other names, rename to ['time', 'length'].
	- Coerce to numeric and drop rows with NaNs in either column.
	- Sort by time ascending and drop duplicate times keeping the first occurrence.
	"""
	original_columns = list(df.columns)
	cols_lower = [str(c).strip().lower() for c in original_columns]

	if {'time', 'length'}.issubset(set(cols_lower)):
		# Map columns to exact names
		col_map = {}
		for c in original_columns:
			cl = str(c).strip().lower()
			if cl == 'time':
				col_map[c] = 'time'
			elif cl == 'length':
				col_map[c] = 'length'
		df = df.rename(columns=col_map)[['time', 'length']]
	else:
		# Fallback: if exactly 2 columns, treat as time, length
		if df.shape[1] != 2:
			raise ValueError(
				f"CSV must contain 'time' and 'length' columns or exactly 2 columns; found columns: {original_columns}"
			)
		df = df.copy()
		df.columns = ['time', 'length']

	# Coerce numeric
	df['time'] = pd.to_numeric(df['time'], errors='coerce')
	df['length'] = pd.to_numeric(df['length'], errors='coerce')
	df = df.dropna(subset=['time', 'length'])

	# Sort and drop duplicates in time
	df = df.sort_values('time', kind='mergesort').drop_duplicates(subset=['time'], keep='first')
	df = df.reset_index(drop=True)
	return df


def load_lt_from_csv() -> pd.DataFrame:
	"""Open a file dialog to select a CSV and load it as a DataFrame with columns ['time', 'length'].

	Returns
	-------
	lt_df : pd.DataFrame
		DataFrame containing two columns: 'time' and 'length'.
	"""
	global _last_selected_csv_path

	# Prefer a GUI dialog when possible; fallback to simple input
	file_path: str | None = None
	if tk is not None and tk_filedialog is not None:
		try:
			root = tk.Tk()
			root.withdraw()
			root.attributes('-topmost', True)
			file_path = tk_filedialog.askopenfilename(
				title='Select length-time CSV',
				filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
			)
			root.destroy()
		except Exception:
			file_path = None

	if not file_path:
		raise RuntimeError('No CSV file selected.')

	if not os.path.isfile(file_path):
		raise FileNotFoundError(f"CSV file not found: {file_path}")

	# Remember for later (plot saving)
	_last_selected_csv_path = file_path

	# Read CSV; be flexible about headers
	try:
		df_try = pd.read_csv(file_path)
	except Exception:
		# Sometimes there is a BOM or encoding issues; retry with utf-8-sig
		df_try = pd.read_csv(file_path, encoding='utf-8-sig')

	lt_df = _ensure_time_length_df(df_try)
	return lt_df


def unify_interval(df_lt: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
	"""Unify the time series to the minimal time interval via linear interpolation.

	Parameters
	----------
	df_lt : pd.DataFrame
		Input DataFrame with columns ['time', 'length'].

	Returns
	-------
	lt_uni_df : pd.DataFrame
		DataFrame reindexed on a uniform time grid with step dt, linearly interpolated lengths.
	dt : float
		The minimal positive time interval found in the original data.
	"""
	if df_lt is None or df_lt.empty:
		raise ValueError('df_lt must be a non-empty DataFrame with columns ["time", "length"].')

	# Ensure clean, sorted, deduplicated input
	df = _ensure_time_length_df(df_lt)

	times = df['time'].to_numpy()
	if times.size < 2:
		raise ValueError('At least two unique time points are required to determine dt.')

	diffs = np.diff(times)
	pos_diffs = diffs[diffs > 0]
	if pos_diffs.size == 0:
		raise ValueError('All time differences are non-positive; cannot determine minimal interval dt.')

	dt = float(np.min(pos_diffs))
	if not np.isfinite(dt) or dt <= 0:
		raise ValueError('Computed dt is not a positive finite number.')

	t_start, t_end = float(times[0]), float(times[-1])
	# Build a uniform grid inclusive of t_end, accounting for floating-point error
	num_steps = int(round((t_end - t_start) / dt))
	# Ensure at least covers the last point
	uniform_times = t_start + dt * np.arange(num_steps + 1, dtype=float)
	if uniform_times[-1] < t_end - 1e-12:
		# Append t_end if needed
		uniform_times = np.append(uniform_times, t_end)

	# Linear interpolation of lengths on the uniform grid
	lengths = df['length'].to_numpy(dtype=float)
	lt_uni = np.interp(uniform_times, times, lengths)

	lt_uni_df = pd.DataFrame({'time': uniform_times, 'length': lt_uni})
	return lt_uni_df, dt


def _normalize_to_unit_range(arr: np.ndarray) -> np.ndarray:
	"""Normalize a 1D array to [-1, 1] via min-max scaling.
	Constant arrays return zeros.
	"""
	if arr.size == 0:
		return arr
	a_min = np.nanmin(arr)
	a_max = np.nanmax(arr)
	if not np.isfinite(a_min) or not np.isfinite(a_max):
		# Fall back to zeros if the feature is entirely NaN/inf
		return np.zeros_like(arr, dtype=float)
	if a_max == a_min:
		return np.zeros_like(arr, dtype=float)
	return 2.0 * (arr - a_min) / (a_max - a_min) - 1.0


def calc_force(df_lt: pd.DataFrame) -> pd.DataFrame:
	"""Compute post deflection-based force for each timepoint and append to DataFrame.

	Steps:
	1) l_mm = mm_per_pixel * length
	2) f = ( 3 * pi * E * (l_base - l_mm) * r^4 ) / ( 4 * post_length^3 )

	Notes:
	- Uses constants measured/calibrated for the setup; adjust if needed.
	- Returns a copy of the input DataFrame (coerced to ['time','length']) with
	  added columns: 'l_mm' and 'force'.
	"""
	if df_lt is None or df_lt.empty:
		raise ValueError('df_lt must be a non-empty DataFrame with columns ["time", "length"].')
	# Preserve existing columns (features, status, etc.). Only validate presence of required columns.
	if 'length' not in df_lt.columns:
		raise ValueError("Column 'length' is required to compute force.")
	df = df_lt.copy()

	# Constants (setup-dependent)
	elas_modulus = 1_700_000.0  # Young's modulus of PDMS (Pa)
	radius = 0.0005             # Radius of PDMS post (m)
	post_length = 0.01          # Length of PDMS post (m)
	l_base = 7.5                # Base center-to-center distance without EHT (mm)
	mm_per_pixel = 0.02       # Calibration factor (mm/pixel); measure per video

	# 1) Convert measured length (pixels) to millimeters
	l_mm = mm_per_pixel * pd.to_numeric(df['length'], errors='coerce').to_numpy(dtype=float)

	# 2) Force calculation per user's formula
	f = (3.0 * math.pi * elas_modulus * (l_base - l_mm) * (radius ** 4)) / (4.0 * (post_length ** 3))

	df['l_mm'] = l_mm
	df['force'] = f
	return df


def calc_features(lt_uni_df: pd.DataFrame, dt: float) -> pd.DataFrame:
	"""Compute velocity (left/right/avg), acceleration, and length deviation features.

	Features are defined on the uniform grid as:
		- v_left  = [l(t) - l(t-1)] / dt
		- v_right = [l(t+1) - l(t)] / dt
		- v       = (v_left + v_right) / 2
		- a       = (v_right - v_left) / dt
		- l_dev   = l(t) - mean(l)

	Normalizes each feature to [-1, 1] and SAVES ONLY the normalized versions
	in the returned DataFrame, without any "_n" suffixes. Original 'time' and
	'length' columns are preserved unchanged.

	Returns a copy of the input DataFrame with columns ['time','length','v_left','v_right','v','a','l_dev']
	where the feature columns are normalized to [-1,1].
	"""
	if lt_uni_df is None or lt_uni_df.empty:
		raise ValueError('lt_uni_df must be a non-empty DataFrame with columns ["time", "length"].')
	if not (isinstance(dt, (int, float)) and dt > 0):
		raise ValueError('dt must be a positive number.')

	df = _ensure_time_length_df(lt_uni_df)
	l = df['length'].to_numpy(dtype=float)
	n = l.size

	v_left = np.empty(n, dtype=float)
	v_right = np.empty(n, dtype=float)

	# Backward difference for v_left, forward difference for v_right
	v_left[0] = np.nan  # placeholder; will fix boundaries below
	v_left[1:] = (l[1:] - l[:-1]) / dt
	v_right[:-1] = (l[1:] - l[:-1]) / dt
	v_right[-1] = np.nan  # placeholder; will fix boundaries below

	# Handle boundaries by mirroring available one-sided difference
	if n >= 2:
		if np.isnan(v_left[0]):
			v_left[0] = v_right[0]
		if np.isnan(v_right[-1]):
			v_right[-1] = v_left[-1]

	v = 0.5 * (v_left + v_right)
	a = (v_right - v_left) / dt

	l_mean = float(np.mean(l)) if n > 0 else 0.0
	l_dev = l - l_mean

	# Normalize features to [-1, 1] and keep ONLY the normalized values (no suffix)
	v_left_n = _normalize_to_unit_range(v_left)
	v_right_n = _normalize_to_unit_range(v_right)
	v_n = _normalize_to_unit_range(v)
	a_n = _normalize_to_unit_range(a)
	l_dev_n = _normalize_to_unit_range(l_dev)

	lt_feat_df = df.copy()
	lt_feat_df['v_left'] = v_left_n
	lt_feat_df['v_right'] = v_right_n
	lt_feat_df['v'] = v_n
	lt_feat_df['a'] = a_n
	lt_feat_df['l_dev'] = l_dev_n

	return lt_feat_df


def classify_states(force_df: pd.DataFrame, dt: float) -> pd.DataFrame:
	"""Single-pass force-derivative state classifier for EHT contraction cycles.

	Uses Savitzky-Golay smoothing, derivative sign, adaptive baseline detection
	(KDE mode), and contraction->relaxation transition peak detection.

	States: 'rest', 'contraction', 'peak', 'relaxation', 'burst rest'

	Parameters
	----------
	force_df : pd.DataFrame with columns ['time', 'force']
	dt : float, sampling interval in seconds

	Returns
	-------
	pd.DataFrame with added 'status' column
	"""
	if force_df is None or force_df.empty:
		raise ValueError('force_df must be a non-empty DataFrame.')
	for col in ('time', 'force'):
		if col not in force_df.columns:
			raise ValueError(f"Column '{col}' is required in force_df.")

	times = force_df['time'].to_numpy(dtype=float)
	forces = force_df['force'].to_numpy(dtype=float)
	n = len(forces)

	# Constants (mirrors LengthDataAnalyzer)
	sg_window_sec = 0.05
	sg_order = 2
	df_threshold_frac = 0.015
	f_rest_threshold = 0.05
	min_segment_samples = 3
	rest_timeout_sec = 0.2
	burst_max_interval_sec = 1.5

	# ---- Phase 1: Smooth and derivative ----
	sg_window = max(3, int(sg_window_sec / dt))
	if sg_window % 2 == 0:
		sg_window += 1
	if n > sg_window:
		f_smooth = savgol_filter(forces, sg_window, sg_order)
	else:
		f_smooth = forces.copy()
	df_dt = np.gradient(f_smooth, dt)

	# ---- Phase 2: Baseline via KDE mode ----
	try:
		kde = gaussian_kde(forces)
		x_grid = np.linspace(float(np.min(forces)), float(np.max(forces)), 500)
		baseline = float(x_grid[np.argmax(kde(x_grid))])
	except Exception:
		hist, edges = np.histogram(forces, bins=min(100, n // 10))
		baseline = float((edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2.0)

	f_range = float(np.max(forces) - np.min(forces))
	if f_range < 1e-12:
		result = force_df.copy()
		result['status'] = 'rest'
		return result

	# ---- Phase 3: Noise estimation ----
	noise_window = 0.15 * f_range
	near_baseline_mask = np.abs(forces - baseline) < noise_window
	if near_baseline_mask.sum() > 20:
		base_df = df_dt[near_baseline_mask]
		mad = float(np.median(np.abs(base_df - np.median(base_df))))
		noise_std = mad * 1.4826
	else:
		noise_std = float(np.std(df_dt))

	df_threshold = max(
		df_threshold_frac * f_range / dt,
		min(3.0 * noise_std, 0.5 * df_threshold_frac * f_range / dt),
	)
	rest_margin = f_rest_threshold * f_range

	# ---- Phase 4: State assignment by derivative ----
	status = np.full(n, '', dtype=object)
	for i in range(n):
		if df_dt[i] > df_threshold:
			status[i] = 'contraction'
		elif df_dt[i] < -df_threshold:
			status[i] = 'relaxation'
		elif np.abs(forces[i] - baseline) <= rest_margin:
			status[i] = 'rest'
		else:
			status[i] = 'rest'

	# ---- Phase 5: Peak detection at contraction->relaxation boundary ----
	is_peak = np.zeros(n, dtype=bool)
	i = 1
	while i < n:
		if status[i-1] == 'contraction' and status[i] in ('relaxation', 'rest'):
			relax_start = i
			while relax_start < n and status[relax_start] == 'rest':
				relax_start += 1
			if relax_start >= n or status[relax_start] != 'relaxation':
				i = relax_start
				continue
			gap_len = relax_start - i
			if gap_len > 3:
				i = relax_start
				continue
			search_start = max(0, i - 1)
			while search_start > 0 and status[search_start] == 'contraction':
				search_start -= 1
			search_end = min(n, relax_start + 1)
			best_j = search_start + int(np.argmax(forces[search_start:search_end]))
			if forces[best_j] >= baseline + 0.05 * f_range:
				is_peak[best_j] = True
			i = relax_start + 1
		else:
			i += 1

	for i in range(n):
		if is_peak[i]:
			status[i] = 'peak'

	all_peak_idx = np.flatnonzero(is_peak)
	all_peak_idx.sort()

	# ---- Phase 6: Post-processing ----
	# Merge short segments (never absorb peaks)
	status = _merge_short_segments(status, min_segment_samples)

	# Burst rest
	if all_peak_idx.size >= 2:
		all_peak_times = times[all_peak_idx]
		for p_idx in range(len(all_peak_idx) - 1):
			p_time = all_peak_times[p_idx]
			next_time = all_peak_times[p_idx + 1]
			if next_time - p_time < burst_max_interval_sec:
				mask_between = (times > p_time) & (times < next_time)
				rest_mask = (status == 'rest')
				label_mask = mask_between & rest_mask
				if np.any(label_mask):
					status[label_mask] = 'burst rest'

	# Cut off stale relaxation
	status = _cutoff_stale_relaxation(times, forces, status, baseline, rest_margin, rest_timeout_sec, dt)

	result = force_df.copy()
	result['status'] = status
	return result


def _merge_short_segments(status: np.ndarray, min_samples: int) -> np.ndarray:
	"""Absorb segments shorter than min_samples into surrounding state.
	Peak segments are never absorbed."""
	n = len(status)
	if n < min_samples:
		return status
	out = status.copy()
	i = 0
	while i < n:
		start = i
		current = out[i]
		while i < n and out[i] == current:
			i += 1
		end = i
		length = end - start
		if length < min_samples and current != 'peak':
			if start > 0:
				replacement = out[start - 1]
			elif end < n:
				replacement = out[end]
			else:
				replacement = 'rest'
			out[start:end] = replacement
	return out


def _cutoff_stale_relaxation(
	times: np.ndarray, forces: np.ndarray, status: np.ndarray,
	baseline: float, rest_margin: float, timeout_sec: float, dt: float,
) -> np.ndarray:
	"""Force relaxation -> rest when force has been near baseline beyond timeout."""
	n = len(status)
	out = status.copy()
	timeout_samples = int(timeout_sec / dt)
	if timeout_samples < 1:
		return out
	i = 0
	while i < n:
		if out[i] == 'relaxation':
			relax_start = i
			while i < n and out[i] == 'relaxation':
				i += 1
			relax_end = i
			seg_f = forces[relax_start:relax_end]
			near_base = np.abs(seg_f - baseline) <= rest_margin
			if np.any(near_base):
				first_base = int(np.argmax(near_base))
				cutoff = relax_start + first_base + timeout_samples
				if cutoff < relax_end:
					out[cutoff:relax_end] = 'rest'
		else:
			i += 1
	return out


def plot_lt_feat(lt_feat_df: pd.DataFrame) -> str:
	"""Create and save a two-part plot for length-time and normalized features.

	The top subplot is the length-time line plot with an adaptive pixel width that
	targets approximately one-pixel-wide space between neighboring points. The
	bottom subplot shows v (orange), a (red), and l_dev (green).

	The figure is not shown; it is saved to the same directory as the last selected
	CSV (from load_lt_from_csv). If no selection is available, it is saved to the
	current working directory. Returns the saved image path.
	"""
	global _last_selected_csv_path

	if lt_feat_df is None or lt_feat_df.empty:
		raise ValueError('lt_feat_df must be a non-empty DataFrame with computed feature columns.')

	df = _ensure_time_length_df(lt_feat_df)
	n = len(df)

	# Choose DPI and width aiming for ~1 pixel gap between adjacent points
	# Approximate: allocate ~2 px per sample (1 px dot + 1 px space)
	dpi = 100
	min_width_px = 800  # sensible minimum
	target_width_px = max(int(2 * n), min_width_px)
	# Cap extremely large figures to avoid memory issues
	max_width_px = 12000
	width_px = min(target_width_px, max_width_px)
	width_in = width_px / dpi

	height_in = 6.0  # total height in inches; split between two subplots

	# Increase height to accommodate 4 stacked plots (length + 3 features)
	height_in = 8.0
	fig, axes = plt.subplots(
		4, 1, figsize=(width_in, height_in), dpi=dpi, sharex=True,
		gridspec_kw={'height_ratios': [2, 1, 1, 1], 'hspace': 0.1}
	)
	ax1, ax2, ax3, ax4 = axes

	# Top: length-time
	ax1.plot(df['time'].to_numpy(), df['length'].to_numpy(), color='tab:blue', linewidth=1.0)
	ax1.set_ylabel('Length')
	ax1.grid(True, linestyle='--', alpha=0.3)

	# Bottom: three separate feature plots using NORMALIZED features stored in lt_feat_df
	time_vals = df['time'].to_numpy()
	v_vals = lt_feat_df.get('v', pd.Series(np.zeros(n))).to_numpy()
	a_vals = lt_feat_df.get('a', pd.Series(np.zeros(n))).to_numpy()
	l_dev_vals = lt_feat_df.get('l_dev', pd.Series(np.zeros(n))).to_numpy()

	# v plot
	ax2.plot(time_vals, v_vals, color='orange')
	ax2.set_ylabel('v')
	ax2.grid(True, linestyle='--', alpha=0.3)

	# a plot
	ax3.plot(time_vals, a_vals, color='red')
	ax3.set_ylabel('a')
	ax3.grid(True, linestyle='--', alpha=0.3)

	# l_dev plot
	ax4.plot(time_vals, l_dev_vals, color='green')
	ax4.set_xlabel('Time')
	ax4.set_ylabel('l_dev')
	ax4.grid(True, linestyle='--', alpha=0.3)

	# Build save path
	if _last_selected_csv_path and os.path.isfile(_last_selected_csv_path):
		base_dir = os.path.dirname(_last_selected_csv_path)
		base_name = os.path.splitext(os.path.basename(_last_selected_csv_path))[0]
	else:
		base_dir = os.getcwd()
		base_name = 'lt_features'

	out_path = os.path.join(base_dir, f"{base_name}_lt_feat.svg")
	fig.savefig(out_path, bbox_inches='tight')
	plt.close(fig)
	return out_path


def plot_force_status(df: pd.DataFrame) -> str:
	"""Plot force-time with status-colored spans.

	- Background spans indicate 'contraction', 'relaxation', and 'burst rest' intervals.
	- The force-time curve is drawn above the spans.
	- Peaks (status == 'peak') are marked as small red dots.
	- X-axis starts at 0 and ends at the last time point.
	- Width adapts to approximately 1 pixel per sample.
	"""
	global _last_selected_csv_path
	if df is None or df.empty:
		raise ValueError('df must be a non-empty DataFrame.')
	for col in ('time', 'force', 'status'):
		if col not in df.columns:
			raise ValueError(f"Column '{col}' is required in df for plotting force identified.")

	# Sort by time to ensure monotonic x for span coloring
	df_sorted = df.sort_values('time', kind='mergesort').reset_index(drop=True)

	# Time series length determines width
	n = len(df_sorted)
	dpi = 100
	min_width_px = 800
	target_width_px = max(int(1 * n), min_width_px)
	max_width_px = 12000
	width_px = min(target_width_px, max_width_px)
	width_in = width_px / dpi
	height_in = 3.0

	fig, ax = plt.subplots(1, 1, figsize=(width_in, height_in), dpi=dpi)

	# Background coloring based on status for each interval [t[i], t[i+1])
	times = df_sorted['time'].to_numpy()
	statuses = df_sorted['status'].to_numpy(dtype=object)
	for i in range(n - 1):
		c = None
		st = statuses[i]
		if st == 'contraction':
			c = '#FF701E'
		elif st == 'relaxation':
			c = '#3FFF0D'
		elif st == 'burst rest':
			c = '#09F8C0'
		if c is not None:
			ax.axvspan(times[i], times[i + 1], facecolor=c, edgecolor='none', alpha=0.25, zorder=0)

	# Plot the force-time line above the background
	ax.plot(times, df_sorted['force'].to_numpy(), color='tab:purple', linewidth=1.0, zorder=2)
	ax.set_xlabel('Time')
	ax.set_ylabel('Force')
	ax.grid(True, linestyle='--', alpha=0.3)
	# Set x-axis to start at 0 and end at the last timepoint
	if n > 0:
		ax.set_xlim(0.0, float(times[-1]))

	# Overlay peaks as ~1-pixel dots
	peaks_mask = df_sorted['status'] == 'peak'
	if peaks_mask.any():
		px_times = np.asarray(df_sorted.loc[peaks_mask, 'time'].values, dtype=float)
		px_force = np.asarray(df_sorted.loc[peaks_mask, 'force'].values, dtype=float)
		# Using scatter with size as area in points^2. Set desired diameter to 2 pixels.
		D_px = 4.0
		D_pt = D_px * 72.0 / dpi  # convert pixels to points based on figure DPI
		size_points2 = math.pi * (D_pt / 2.0) ** 2
		ax.scatter(px_times, px_force, s=size_points2, c='red', marker='o', linewidths=0, zorder=5)

	# Save path
	if _last_selected_csv_path and os.path.isfile(_last_selected_csv_path):
		base_dir = os.path.dirname(_last_selected_csv_path)
		base_name = os.path.splitext(os.path.basename(_last_selected_csv_path))[0]
	else:
		base_dir = os.getcwd()
		base_name = 'force_status'

	out_path = os.path.join(base_dir, f"{base_name}_force_status.svg")
	fig.savefig(out_path, bbox_inches='tight')
	plt.close(fig)
	return out_path



if __name__ == "__main__":
	# 1) Pick CSV and load it
	lt_df = load_lt_from_csv()  # select a CSV with columns time,length

	# 2) Make time uniform at the minimal interval
	lt_uni_df, dt = unify_interval(lt_df)

	# 3) Compute features
	lt_feat_df = calc_features(lt_uni_df, dt)
 
	# 4) Calculate force
	lt_feat_df = calc_force(lt_feat_df)

	# 5) Classify states using force-derivative method
	lt_identified_df = classify_states(lt_feat_df, dt)

	# 7) Plot force/status
	out_force_status_path = plot_force_status(lt_identified_df)
	print(f"Saved force status plot to: {out_force_status_path}")

	# 8) Save identified DataFrame to CSV
	if _last_selected_csv_path and os.path.isfile(_last_selected_csv_path):
		base_dir = os.path.dirname(_last_selected_csv_path)
		base_name = os.path.splitext(os.path.basename(_last_selected_csv_path))[0]
		out_csv_path = os.path.join(base_dir, f"{base_name}_identified.csv")
		lt_identified_df.to_csv(out_csv_path, index=False)
		print(f"Saved identified DataFrame to CSV: {out_csv_path}")