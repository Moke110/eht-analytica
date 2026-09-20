from __future__ import annotations

import os
import re
import math
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde


class LengthDataAnalyzer:
	"""End-to-end analysis pipeline for length-time data to force/status and summary metrics.

	Responsibilities:
	- Load a length-time CSV and standardize columns to ['time','length'].
	- Unify to minimal time interval via linear interpolation.
	- Compute features (v_left, v_right, v, a, l_dev) normalized to [-1, 1].
	- Convert length to force using calibrated constants.
	- Identify peaks and movement spans (contraction/relaxation).
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

	# Force-derivative classifier constants
	SG_WINDOW_SEC = 0.05          # Savitzky-Golay filter window (seconds)
	SG_ORDER = 2                  # Savitzky-Golay polynomial order
	PEAK_PROMINENCE_FRAC = 0.10   # Min peak prominence as fraction of force range
	PEAK_MIN_DIST_SEC = 0.08      # Min distance between peaks (seconds)
	DF_THRESHOLD_FRAC = 0.015      # Derivative threshold as fraction of (force_range / dt)
	F_REST_THRESHOLD = 0.05       # Force within 5% of baseline range → candidate rest
	MIN_SEGMENT_SAMPLES = 5       # Minimum consecutive samples to keep a state segment
	REST_TIMEOUT_SEC = 0.2        # Max time after relaxation reaches baseline before forcing rest
	MIN_FORCE_RANGE = 0.025       # Minimum force range (N) to consider as active contraction
	MIN_DIASTOLIC_SEGMENTS = 3   # Minimum relaxation segments to compute diastolic interval

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

	def classify_states(self, force_df: pd.DataFrame, dt: float) -> pd.DataFrame:
		"""Single-pass force-derivative state classifier.

		Uses Savitzky-Golay smoothing, derivative sign, adaptive baseline detection
		(KDE mode), and contraction->relaxation transition peak detection with
		local-maximum prominence validation.

		States: 'rest', 'contraction', 'peak', 'relaxation'
		"""
		if force_df is None or force_df.empty:
			raise ValueError('force_df must be a non-empty DataFrame.')
		for col in ('time', 'force'):
			if col not in force_df.columns:
				raise ValueError(f"Column '{col}' is required in force_df.")

		times = force_df['time'].to_numpy(dtype=float)
		forces = force_df['force'].to_numpy(dtype=float)
		n = len(forces)

		# ---- Phase 1: Smooth force and compute derivative ----
		sg_window = max(3, int(self.SG_WINDOW_SEC / dt))
		if sg_window % 2 == 0:
			sg_window += 1
		if n > sg_window:
			f_smooth = savgol_filter(forces, sg_window, self.SG_ORDER)
		else:
			f_smooth = forces.copy()
		df_dt = np.gradient(f_smooth, dt)

		# ---- Phase 2: Adaptive baseline via KDE mode ----
		try:
			kde = gaussian_kde(forces)
			x_grid = np.linspace(float(np.min(forces)), float(np.max(forces)), 500)
			baseline = float(x_grid[np.argmax(kde(x_grid))])
		except Exception:
			hist, edges = np.histogram(forces, bins=min(100, n // 10))
			baseline = float((edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2.0)

		f_range = float(np.max(forces) - np.min(forces))
		if f_range < self.MIN_FORCE_RANGE:
			result = force_df.copy()
			result['status'] = 'rest'
			return result

		# ---- Phase 3: Derivative noise estimation (MAD-based) ----
		noise_window = 0.15 * f_range
		near_baseline_mask = np.abs(forces - baseline) < noise_window
		if near_baseline_mask.sum() > 20:
			base_df = df_dt[near_baseline_mask]
			mad = float(np.median(np.abs(base_df - np.median(base_df))))
			noise_std = mad * 1.4826  # MAD -> std for normal distribution
		else:
			noise_std = float(np.std(df_dt))

		# Adaptive derivative threshold (capped noise floor)
		df_threshold = max(
			self.DF_THRESHOLD_FRAC * f_range / dt,
			min(3.0 * noise_std, 0.5 * self.DF_THRESHOLD_FRAC * f_range / dt),
		)
		rest_margin = self.F_REST_THRESHOLD * f_range

		# ---- Phase 4: State assignment by derivative sign ----
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

		# ---- Phase 4.5: Diastolic interval override ----
		# Compute the diastolic force range from relaxation segment minima and
		# force all points whose force falls within it to 'rest', regardless of
		# derivative sign.  This suppresses noise-induced misclassification in
		# the diastolic (relaxed) band.
		diastolic_lower, diastolic_upper = self._compute_diastolic_interval(
			status, forces, baseline, f_range, rest_margin,
		)
		diastolic_mask = (forces >= diastolic_lower) & (forces <= diastolic_upper)
		status[diastolic_mask] = 'rest'

		# ---- Phase 4b: Merge short segments BEFORE peak detection ----
		# Absorb noise-induced tiny contraction/relaxation segments into the
		# surrounding state so they don't produce spurious peaks.
		status = self._merge_short_segments(status, self.MIN_SEGMENT_SAMPLES)

		# ---- Phase 5: Peak detection at contraction->relaxation boundary ----
		# Place exactly one peak per contraction-relaxation transition, then skip
		# the entire relaxation segment + a minimum distance.  Each candidate is
		# validated for local-maximum prominence to filter out noisy zero-crossings.
		is_peak = np.zeros(n, dtype=bool)
		peak_min_dist_samples = max(1, int(self.PEAK_MIN_DIST_SEC / dt))
		min_prominence = self.PEAK_PROMINENCE_FRAC * f_range
		prominence_window = max(2, int(0.10 / dt))  # ~100 ms each side
		last_peak_idx = -peak_min_dist_samples
		i = 1
		while i < n:
			if status[i - 1] == 'contraction' and status[i] in ('relaxation', 'rest'):
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

				# Locate the force maximum in the transition window
				search_start = max(0, i - 1)
				while search_start > 0 and status[search_start] == 'contraction':
					search_start -= 1
				search_end = min(n, relax_start + 1)
				best_j = search_start + int(np.argmax(forces[search_start:search_end]))

				# ---- Validate candidate peak ----
				valid = False
				if forces[best_j] >= baseline + 0.05 * f_range and (best_j - last_peak_idx) >= peak_min_dist_samples:
					# Local-maximum check
					vl = max(0, best_j - prominence_window)
					vr = min(n, best_j + prominence_window + 1)
					if forces[best_j] >= np.max(forces[vl:vr]):
						# Prominence check: force drop on both sides
						left_min = float(np.min(forces[vl:best_j])) if best_j > vl else forces[best_j]
						right_min = float(np.min(forces[best_j:vr])) if vr > best_j else forces[best_j]
						if (forces[best_j] - left_min) >= min_prominence and (forces[best_j] - right_min) >= min_prominence:
							valid = True

				if valid:
					is_peak[best_j] = True
					last_peak_idx = best_j

				# Skip the entire relaxation segment + enforce minimum peak distance
				i = relax_start
				while i < n and status[i] == 'relaxation':
					i += 1
				i = max(i, last_peak_idx + peak_min_dist_samples)
			else:
				i += 1

		for j in range(n):
			if is_peak[j]:
				status[j] = 'peak'

		# ---- Phase 6: Post-processing ----
		# Cut off stale relaxation
		status = self._cutoff_stale_relaxation(times, forces, status, baseline,
		                                       rest_margin, self.REST_TIMEOUT_SEC, dt)

		result = force_df.copy()
		result['status'] = status
		return result


	@staticmethod
	def _compute_diastolic_interval(
		status: np.ndarray,
		forces: np.ndarray,
		baseline: float,
		f_range: float,
		rest_margin: float,
		min_samples: int = 5,
		min_relax_segments: int = 3,
	) -> tuple[float, float]:
		"""Compute the diastolic force interval from relaxation-segment minima.

		1. Scan all 'relaxation' segments.
		2. For each segment with length >= min_samples, collect its minimum force.
		3. If fewer than min_relax_segments valid segments exist, fall back to
		   [baseline - rest_margin, baseline + rest_margin].
		4. Otherwise, centre the interval on *baseline* with a width derived from
		   how far relaxation minima deviate from baseline, capped at 10 % of the
		   total force range to avoid eating into genuine contractions.

		Returns (diastolic_lower, diastolic_upper).
		"""
		n = len(status)

		# ---- Collect minima from qualifying relaxation segments ----
		mins: list[float] = []
		i = 0
		while i < n:
			if status[i] == 'relaxation':
				seg_start = i
				while i < n and status[i] == 'relaxation':
					i += 1
				seg_len = i - seg_start
				if seg_len >= min_samples:
					mins.append(float(np.min(forces[seg_start:i])))
			else:
				i += 1

		# ---- Fallback when not enough valid segments ----
		if len(mins) < min_relax_segments:
			return (baseline - rest_margin, baseline + rest_margin)

		# ---- Baseline-centred width from relaxation-minima deviations ----
		mins_arr = np.array(mins)
		deviations = np.abs(mins_arr - baseline)
		median_dev = float(np.median(deviations))
		# Width is at least rest_margin, at most 10 % of total force range
		width = max(rest_margin, 2.0 * median_dev)
		width = min(width, 0.10 * f_range)

		diastolic_lower = baseline - width
		diastolic_upper = baseline + width
		return (diastolic_lower, diastolic_upper)


	@staticmethod
	def _merge_short_segments(status: np.ndarray, min_samples: int) -> np.ndarray:
		"""Absorb segments shorter than min_samples into surrounding state.

		Peak segments are never absorbed — they are preserved regardless of length.
		"""
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
				# Determine replacement from neighbours
				if start > 0 and end < n:
					# Use the preceding neighbour if it exists
					replacement = out[start - 1]
				elif start > 0:
					replacement = out[start - 1]
				elif end < n:
					replacement = out[end]
				else:
					replacement = 'rest'
				out[start:end] = replacement
		return out

	@staticmethod
	def _cutoff_stale_relaxation(
		times: np.ndarray, forces: np.ndarray, status: np.ndarray,
		baseline: float, rest_margin: float, timeout_sec: float, dt: float,
	) -> np.ndarray:
		"""Force relaxation → rest when force has been near baseline beyond timeout."""
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
	) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
		"""Run the full workflow on input length-time CSV paths and return in-memory results only.

		New simplified workflow (per latest user request):
		- No filesystem outputs are produced.
		- For each input CSV named pattern ``{EHT_name}_length.csv`` (fallback to entire basename if pattern not matched):
		  * Process through unify -> features -> force -> peak/movement identification.
		  * Collect a time-force-status DataFrame with columns ['time','force','status'].
		- Aggregate metrics across samples and return a single metrics DataFrame with one row per input.

		Parameters:
		- csv_paths: list of CSV file paths.

		Returns:
		- (list_of_force_status_dfs, metrics_df)
		"""
		if csv_paths is None or len(csv_paths) == 0:
			raise ValueError('csv_paths must be a non-empty list of file paths.')

		force_status_dfs: List[pd.DataFrame] = []
		metrics_rows: List[pd.DataFrame] = []

		for csv_path in csv_paths:
			if not os.path.isfile(csv_path):
				raise FileNotFoundError(f"CSV not found: {csv_path}")
			base_name = os.path.splitext(os.path.basename(csv_path))[0]
			# New naming convention: {EHT_name}_length.csv -> extract EHT_name
			m_length = re.match(r"^(?P<eht>.+)_length$", base_name)
			if m_length:
				name = m_length.group('eht')
			else:
				# Fallback: previous pattern {EHT_name}_tracked or use full base_name
				m_tracked = re.match(r"^(?P<eht>.+)_tracked$", base_name)
				name = m_tracked.group('eht') if m_tracked else base_name

			# Load, unify, calc force (features kept for calc_force dependency)
			df_raw = pd.read_csv(csv_path)
			lt_df = self._ensure_time_length_df(df_raw)
			lt_uni_df, dt = self.unify_interval(lt_df)
			feat_df = self.calc_features(lt_uni_df, dt)
			force_df = self.calc_force(feat_df)
			identified_df = self.classify_states(force_df, dt)

			# Collect force/status per sample (in memory only)
			force_status_df = identified_df[['time', 'force', 'status']].copy()
			force_status_df.attrs['EHT_name'] = name
			force_status_dfs.append(force_status_df)

			# Compute metrics row per sample
			cycles_df = self.seg_cycles(identified_df)
			analysis_df = self.analyze_data(identified_df, cycles_df)
			analysis_df.loc[0, 'EHT name'] = name
			metrics_rows.append(analysis_df)

		metrics_df = pd.concat(metrics_rows, ignore_index=True) if metrics_rows else pd.DataFrame()
		return force_status_dfs, metrics_df
