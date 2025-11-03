from __future__ import annotations

import os
import re
import math
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd


class LengthDataAnalyzer:
	"""End-to-end analysis pipeline for length-time data to force/status and summary metrics.

	Responsibilities:
	- Load a length-time CSV and standardize columns to ['time','length'].
	- Unify to minimal time interval via linear interpolation.
	- Compute features (v_left, v_right, v, a, l_dev) normalized to [-1, 1].
	- Convert length to force using calibrated constants.
	- Identify peaks and movement spans (contraction/relaxation) with burst rest labeling.
	- Segment cycles and compute t80 metrics.
	- Analyze dataset-level metrics and return/save results.

	Supports:
	- Multi-CSV workflow via analyze_lt_csvs(...), which takes a list of N CSV paths and writes
	  N force/status CSVs plus one metrics CSV (one row per input file).

	This class intentionally excludes visualization; only CSV outputs are written.
	"""

	# Calibration / mechanical constants
	ELAS_MODULUS_PA = 1_700_000.0  # Young's modulus of PDMS (Pa)
	POST_RADIUS_M = 0.0005         # Radius of PDMS post (m)
	POST_LENGTH_M = 0.01           # Length of PDMS post (m)
	L_BASE_MM = 7.5                # Base distance (mm)
	MM_PER_PIXEL = 0.02            # Calibration factor (mm/pixel), measure per video

	# Movement labeling thresholds (mirrors test move_identifier implementation)
	CONTRACTION_STOP_VRIGHT = -0.03
	RELAXATION_STOP_VLEFT = 0.03
	STOP_LDEV = 0.25
	BURST_INTERVAL_PERCENTILE = 75.0
	BURST_WINDOW_FACTOR = 1.5  # window = 1.5 * burst_peak_interval

	def _ensure_time_length_df(self, df: pd.DataFrame) -> pd.DataFrame:
		original_columns = list(df.columns)
		cols_lower = [str(c).strip().lower() for c in original_columns]
		if {'time', 'length'}.issubset(set(cols_lower)):
			col_map: dict[str, str] = {}
			for c in original_columns:
				cl = str(c).strip().lower()
				if cl == 'time':
					col_map[c] = 'time'
				elif cl == 'length':
					col_map[c] = 'length'
			out = df.rename(columns=col_map)[['time', 'length']]
		else:
			if df.shape[1] != 2:
				raise ValueError(
					f"CSV must contain 'time' and 'length' columns or exactly 2 columns; found columns: {original_columns}"
				)
			out = df.copy()
			out.columns = ['time', 'length']

		out['time'] = pd.to_numeric(out['time'], errors='coerce')
		out['length'] = pd.to_numeric(out['length'], errors='coerce')
		out = out.dropna(subset=['time', 'length'])
		out = out.sort_values('time', kind='mergesort').drop_duplicates(subset=['time'], keep='first')
		return out.reset_index(drop=True)

	def unify_interval(self, df_lt: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
		if df_lt is None or df_lt.empty:
			raise ValueError('df_lt must be a non-empty DataFrame with columns ["time", "length"].')
		df = self._ensure_time_length_df(df_lt)
		times = df['time'].to_numpy()
		if times.size < 2:
			raise ValueError('At least two unique time points are required to determine dt.')
		diffs = np.diff(times)
		pos_diffs = diffs[diffs > 0]
		if pos_diffs.size == 0:
			raise ValueError('All time differences are non-positive; cannot determine minimal interval dt.')
		dt = float(np.min(pos_diffs))
		t_start, t_end = float(times[0]), float(times[-1])
		num_steps = int(round((t_end - t_start) / dt))
		uniform_times = t_start + dt * np.arange(num_steps + 1, dtype=float)
		if uniform_times[-1] < t_end - 1e-12:
			uniform_times = np.append(uniform_times, t_end)
		lengths = df['length'].to_numpy(dtype=float)
		lt_uni = np.interp(uniform_times, times, lengths)
		return pd.DataFrame({'time': uniform_times, 'length': lt_uni}), dt

	@staticmethod
	def _normalize_to_unit_range(arr: np.ndarray) -> np.ndarray:
		if arr.size == 0:
			return arr
		a_min = np.nanmin(arr)
		a_max = np.nanmax(arr)
		if not np.isfinite(a_min) or not np.isfinite(a_max) or a_max == a_min:
			return np.zeros_like(arr, dtype=float)
		return 2.0 * (arr - a_min) / (a_max - a_min) - 1.0

	def calc_features(self, lt_uni_df: pd.DataFrame, dt: float) -> pd.DataFrame:
		if lt_uni_df is None or lt_uni_df.empty:
			raise ValueError('lt_uni_df must be a non-empty DataFrame with columns ["time", "length"].')
		if not (isinstance(dt, (int, float)) and dt > 0):
			raise ValueError('dt must be a positive number.')
		df = self._ensure_time_length_df(lt_uni_df)
		l = df['length'].to_numpy(dtype=float)
		n = l.size
		v_left = np.empty(n, dtype=float)
		v_right = np.empty(n, dtype=float)
		v_left[0] = np.nan
		v_left[1:] = (l[1:] - l[:-1]) / dt
		v_right[:-1] = (l[1:] - l[:-1]) / dt
		v_right[-1] = np.nan
		if n >= 2:
			if np.isnan(v_left[0]):
				v_left[0] = v_right[0]
			if np.isnan(v_right[-1]):
				v_right[-1] = v_left[-1]
		v = 0.5 * (v_left + v_right)
		a = (v_right - v_left) / dt
		l_mean = float(np.mean(l)) if n > 0 else 0.0
		l_dev = l - l_mean
		lt_feat_df = df.copy()
		lt_feat_df['v_left'] = self._normalize_to_unit_range(v_left)
		lt_feat_df['v_right'] = self._normalize_to_unit_range(v_right)
		lt_feat_df['v'] = self._normalize_to_unit_range(v)
		lt_feat_df['a'] = self._normalize_to_unit_range(a)
		lt_feat_df['l_dev'] = self._normalize_to_unit_range(l_dev)
		return lt_feat_df

	def calc_force(self, df_lt: pd.DataFrame) -> pd.DataFrame:
		if df_lt is None or df_lt.empty:
			raise ValueError('df_lt must be a non-empty DataFrame.')
		if 'length' not in df_lt.columns:
			raise ValueError("Column 'length' is required to compute force.")
		df = df_lt.copy()
		l_mm = self.MM_PER_PIXEL * pd.to_numeric(df['length'], errors='coerce').to_numpy(dtype=float)
		f = (3.0 * math.pi * self.ELAS_MODULUS_PA * (self.L_BASE_MM - l_mm) * (self.POST_RADIUS_M ** 4)) / (4.0 * (self.POST_LENGTH_M ** 3))
		df['l_mm'] = l_mm
		df['force'] = f
		return df

	def identify_peaks(self, lt_feat_df: pd.DataFrame) -> pd.DataFrame:
		if lt_feat_df is None or lt_feat_df.empty:
			raise ValueError('lt_feat_df must be a non-empty DataFrame with features.')
		required_cols = ('time', 'length', 'a', 'l_dev', 'v_left', 'v_right')
		for col in required_cols:
			if col not in lt_feat_df.columns:
				raise ValueError(f"Column '{col}' is required in lt_feat_df.")
		peak_df = lt_feat_df.copy()
		status = np.full(len(peak_df), 'rest', dtype=object)
		mask_peak = (
			(peak_df['l_dev'].to_numpy() < 0.1)
			& (peak_df['a'].to_numpy() > 0)
			& (peak_df['v_left'].to_numpy() < 0)
			& (peak_df['v_right'].to_numpy() > 0)
		)
		status[mask_peak] = 'peak'
		peak_df['status'] = status
		return peak_df

	def identify_move_on_peaks(self, peak_df: pd.DataFrame) -> pd.DataFrame:
		if peak_df is None or peak_df.empty:
			raise ValueError('peak_df must be a non-empty DataFrame.')
		required_cols = ('time', 'status', 'v_right', 'v_left', 'l_dev')
		for col in required_cols:
			if col not in peak_df.columns:
				raise ValueError(f"Column '{col}' is required in peak_df.")
		identified_df = peak_df.copy()
		statuses = identified_df['status'].to_numpy(dtype=object)
		v_right = identified_df['v_right'].to_numpy(dtype=float)
		v_left = identified_df['v_left'].to_numpy(dtype=float)
		l_dev = identified_df['l_dev'].to_numpy(dtype=float)
		times = identified_df['time'].to_numpy(dtype=float)
		n = len(identified_df)

		peak_indices = np.flatnonzero(statuses == 'peak')
		for p in peak_indices:
			# Left: label contraction until stop condition
			i = p - 1
			while i >= 0:
				if (v_right[i] > self.CONTRACTION_STOP_VRIGHT) and (l_dev[i] > self.STOP_LDEV):
					break
				if statuses[i] == 'rest':
					statuses[i] = 'contraction'
				i -= 1
			# Right: label relaxation until stop condition
			i = p + 1
			while i < n:
				if (v_left[i] < self.RELAXATION_STOP_VLEFT) and (l_dev[i] > self.STOP_LDEV):
					break
				if statuses[i] == 'rest':
					statuses[i] = 'relaxation'
				i += 1

		# Burst rest labeling via 75th percentile interval and window factor
		if peak_indices.size >= 2:
			peak_times = times[peak_indices]
			intervals = np.diff(peak_times)
			if intervals.size > 0:
				burst_peak_interval = float(np.percentile(intervals, self.BURST_INTERVAL_PERCENTILE))
			else:
				burst_peak_interval = None
		else:
			burst_peak_interval = None

		if burst_peak_interval is not None and np.isfinite(burst_peak_interval) and burst_peak_interval > 0:
			for p_idx in peak_indices:
				p_time = times[p_idx]
				cand = peak_indices[peak_indices > p_idx]
				if cand.size == 0:
					continue
				cand_times = times[cand]
				window_end = p_time + self.BURST_WINDOW_FACTOR * burst_peak_interval
				in_window = cand[cand_times < window_end]
				if in_window.size == 0:
					continue
				next_peak_idx = int(in_window[0])
				next_time = times[next_peak_idx]
				mask_between = (times > p_time) & (times < next_time)
				rest_mask = (statuses == 'rest')
				label_mask = mask_between & rest_mask
				if np.any(label_mask):
					statuses[label_mask] = 'burst rest'

		identified_df['status'] = statuses
		return identified_df

	def seg_cycles(self, df: pd.DataFrame) -> pd.DataFrame:
		if df is None or df.empty:
			return pd.DataFrame(columns=['time_arr', 'force_arr', 'force_start', 'force_peak', 'force_end', 'time_peak', 't80_con', 't80_rel'])
		for col in ('time', 'force', 'status'):
			if col not in df.columns:
				raise ValueError(f"Column '{col}' is required in df for cycle segmentation.")
		sdf = df.sort_values('time', kind='mergesort').reset_index(drop=True)
		times = sdf['time'].to_numpy(dtype=float)
		forces = sdf['force'].to_numpy(dtype=float)
		statuses = sdf['status'].to_numpy(dtype=object)
		n = len(sdf)
		peak_idxs = np.flatnonzero(statuses == 'peak')
		if peak_idxs.size == 0:
			return pd.DataFrame(columns=['time_arr', 'force_arr', 'force_start', 'force_peak', 'force_end', 'time_peak', 't80_con', 't80_rel'])
		rows = []
		last_end = -1
		for p in peak_idxs:
			i = p - 1
			while i >= 0 and statuses[i] == 'contraction':
				i -= 1
			left = max(i + 1, 0)
			left = max(left, last_end + 1)
			j = p + 1
			while j < n and statuses[j] == 'relaxation':
				j += 1
			right = min(j - 1, n - 1) if j > p + 1 else p
			if right < left:
				continue
			t_arr = times[left:right + 1]
			f_arr = forces[left:right + 1]
			if t_arr.size == 0:
				continue
			local_max_idx = int(np.argmax(f_arr))
			force_peak = float(f_arr[local_max_idx])
			time_peak = float(t_arr[local_max_idx])
			f_start = float(f_arr[0])
			f_end = float(f_arr[-1])
			# t80 rise
			target_con = f_start + 0.2 * (force_peak - f_start)
			t80_con = np.nan
			con_idx = np.argmax(f_arr >= target_con) if f_arr.size > 0 else 0
			if f_arr.size > 0 and (f_arr[con_idx] >= target_con):
				t80_con = float(t_arr[con_idx] - t_arr[0])
			# t80 relax
			target_rel = f_end + 0.2 * (force_peak - f_end)
			t80_rel = np.nan
			if force_peak >= f_end:
				rel_mask = f_arr[local_max_idx:] <= target_rel
			else:
				rel_mask = f_arr[local_max_idx:] >= target_rel
			if rel_mask.size > 0 and np.any(rel_mask):
				rel_offset = int(np.argmax(rel_mask))
				rel_idx = local_max_idx + rel_offset
				t80_rel = float(t_arr[rel_idx] - t_arr[local_max_idx])
			rows.append({
				'time_arr': t_arr.copy(),
				'force_arr': f_arr.copy(),
				'force_start': f_start,
				'force_peak': force_peak,
				'force_end': f_end,
				'time_peak': time_peak,
				't80_con': t80_con,
				't80_rel': t80_rel,
			})
			last_end = right
		return pd.DataFrame(rows, columns=['time_arr', 'force_arr', 'force_start', 'force_peak', 'force_end', 'time_peak', 't80_con', 't80_rel'])

	def analyze_data(self, lt_identified_df: pd.DataFrame, cycles_df: pd.DataFrame) -> pd.DataFrame:
		force_con = float(np.nanmean(cycles_df['force_peak'])) if not cycles_df.empty else np.nan
		if lt_identified_df is not None and not lt_identified_df.empty:
			rest_forces = pd.to_numeric(lt_identified_df.loc[lt_identified_df['status'] == 'rest', 'force'], errors='coerce')
			force_rest = float(np.nanmean(rest_forces)) if rest_forces.size > 0 else np.nan
		else:
			force_rest = np.nan
		parts: list[np.ndarray] = []
		if not cycles_df.empty:
			parts.append(pd.to_numeric(cycles_df['force_start'], errors='coerce').to_numpy())
			parts.append(pd.to_numeric(cycles_df['force_end'], errors='coerce').to_numpy())
		if lt_identified_df is not None and not lt_identified_df.empty:
			burst_forces = pd.to_numeric(lt_identified_df.loc[lt_identified_df['status'] == 'burst rest', 'force'], errors='coerce').to_numpy()
			if burst_forces.size > 0:
				parts.append(burst_forces)
		force_burst_rest = float(np.nanmean(np.concatenate(parts))) if len(parts) > 0 else np.nan
		if lt_identified_df is not None and not lt_identified_df.empty:
			t_min = float(np.nanmin(lt_identified_df['time']))
			t_max = float(np.nanmax(lt_identified_df['time']))
			total_time = t_max - t_min
			n_peaks = int(np.sum(lt_identified_df['status'] == 'peak'))
			freq_bpm = ((n_peaks / total_time) * 60.0) if total_time > 0 else np.nan
		else:
			freq_bpm = np.nan
		t80_con_mean = float(np.nanmean(cycles_df['t80_con'])) if ('t80_con' in cycles_df.columns and not cycles_df.empty) else np.nan
		t80_rel_mean = float(np.nanmean(cycles_df['t80_rel'])) if ('t80_rel' in cycles_df.columns and not cycles_df.empty) else np.nan
		dt = force_burst_rest - force_rest if (np.isfinite(force_burst_rest) and np.isfinite(force_rest)) else np.nan
		return pd.DataFrame([{
			'EHT name': '',
			'Contraction Force': force_con,
			'Relaxation Force': force_rest,
			'Diastolic Tension': dt,
			'Frequency': freq_bpm,
			'Time to Peak 80%': t80_con_mean,
			'Relaxation Time 80%': t80_rel_mean,
		}])

	def analyze_lt_csvs(
		self,
		csv_paths: List[str],
		output_dir: str,
		metrics_filename: str = 'metrics.csv',
	) -> Tuple[List[str], str]:
		"""Run the full workflow on a list of length-time CSV file paths.

		For each input CSV, one force/status CSV is written; additionally, a single
		metrics CSV is written containing one row per input. Returns a tuple of
		(list_of_force_status_paths, metrics_csv_path).

		Parameters:
		- csv_paths: List of CSV file paths; each CSV must contain columns ['time','length']
		            (or exactly two numeric columns convertible to that).
		- output_dir: Directory to write outputs into. Will be created if it doesn't exist.
		- metrics_filename: Filename for the aggregated metrics CSV inside output_dir.

		Returns:
		- (force_identified_paths, metrics_path)
		"""
		if csv_paths is None or len(csv_paths) == 0:
			raise ValueError('csv_paths must be a non-empty list of file paths.')
		os.makedirs(output_dir, exist_ok=True)

		force_identified_paths: List[str] = []
		metrics_rows: List[pd.DataFrame] = []

		for csv_path in csv_paths:
			if not os.path.isfile(csv_path):
				raise FileNotFoundError(f"CSV not found: {csv_path}")
			base_name = os.path.splitext(os.path.basename(csv_path))[0]
			m = re.match(r"^(?P<eht>.+)_tracked$", base_name)
			name = m.group('eht') if m else base_name

			# Load, unify, features, force, identify
			df_raw = pd.read_csv(csv_path)
			lt_df = self._ensure_time_length_df(df_raw)
			lt_uni_df, dt = self.unify_interval(lt_df)
			feat_df = self.calc_features(lt_uni_df, dt)
			force_df = self.calc_force(feat_df)
			peaks_df = self.identify_peaks(force_df)
			identified_df = self.identify_move_on_peaks(peaks_df)

			# Save force/status per sample
			force_identified = identified_df[['time', 'force', 'status']].copy()
			force_identified_path = os.path.join(output_dir, f"{name}_force_identified.csv")
			force_identified.to_csv(force_identified_path, index=False)
			force_identified_paths.append(force_identified_path)

			# Compute metrics row per sample
			cycles_df = self.seg_cycles(identified_df)
			analysis_df = self.analyze_data(identified_df, cycles_df)
			analysis_df.loc[0, 'EHT name'] = name
			metrics_rows.append(analysis_df)

		# Write aggregated metrics
		metrics_df = pd.concat(metrics_rows, ignore_index=True) if metrics_rows else pd.DataFrame()
		metrics_path = os.path.join(output_dir, metrics_filename)
		metrics_df.to_csv(metrics_path, index=False)

		return force_identified_paths, metrics_path

