"""
Tracking module for processing video frames with ROI-based tracking.

This implementation generates a single time-length CSV per recording without
creating annotated frames or tracked videos.
"""

import base64
import cv2
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path


# ── Canvas layout constants (pixels) ─────────────────────────────────────────

TITLE_HEIGHT = 20        # recording name row (16 px bold)
SEP_HEIGHT = 8           # separator bar (pure black)
ROI_DISPLAY_SIZE = 256   # each ROI cell is 256×256
ROI_GAP = 5              # horizontal gap between ROI cells
LABEL_HEIGHT = 14        # ROI name row (14 px)
BORDER = 16              # four-side border around the whole video

# Inner canvas (content area, excluding border)
INNER_HEIGHT = TITLE_HEIGHT + SEP_HEIGHT + ROI_DISPLAY_SIZE + SEP_HEIGHT + LABEL_HEIGHT  # 302
CANVAS_HEIGHT = BORDER * 2 + INNER_HEIGHT  # 334


def _build_canvas_width(num_rois: int) -> int:
    """Total canvas width (including borders) given n ROIs."""
    inner_w = num_rois * ROI_DISPLAY_SIZE + (num_rois + 1) * ROI_GAP
    return BORDER * 2 + inner_w


# ── Tracking ─────────────────────────────────────────────────────────────────

def process_tracking(video_path, rois, infer_fn, output_folder, progress_callback=None,
                     total_frames_hint: int | None = None, frame_buffer: list | None = None,
                     save_tracked_video: bool = False,
                     save_inferences: bool = False,
                     model_name: str | None = None,
                     target_size: int = 256):
    """
    Process video tracking for each ROI and save one length CSV per ROI.

    When *save_tracked_video* is True, a JSON temp file
    ``.{recording}_tracked_frames.json`` is pre-populated with one row per
    expected frame (based on recording fps × duration).  Rows are filled
    during tracking by matching each real frame's timestamp to the closest
    expected row.  The caller should call :func:`compose_tracked_video_from_json`
    afterwards to produce the final frame-aligned video and then delete the
    JSON.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    rois : list of dict
        Each dict: name, x, y, width, height, color, sample_id, recording_name.
    infer_fn : callable
        infer_fn(roi_images: list[np.ndarray]) -> list[np.ndarray]
    output_folder : str
        Path to the job directory (EHT-analytics).
    progress_callback : callable, optional
        Callback(current_frame, total_frames, message).
    total_frames_hint : int, optional
    frame_buffer : list, optional
        Pre-processed frame buffer.
    save_tracked_video : bool
    save_inferences : bool
    model_name : str, optional
    target_size : int
        Model input resolution (256 for v3, 512 for v2). Stored in inference metadata.

    Returns
    -------
    dict
        {'success': bool, 'message': str, 'saved_files': list[str],
         'tracked_frames_json': str | None}
    """

    if not rois:
        return {'success': False, 'message': 'No ROIs defined. Please draw ROIs first.'}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'success': False, 'message': f'Could not open video: {video_path}'}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if isinstance(frame_buffer, list) and len(frame_buffer) > 0:
        progress_total = len(frame_buffer)
    else:
        progress_total = total_frames_hint if (isinstance(total_frames_hint, int) and total_frames_hint > 0) else total_frames

    lengths_dir = os.path.join(output_folder, "lengths")
    os.makedirs(lengths_dir, exist_ok=True)
    recording_name = os.path.splitext(os.path.basename(video_path))[0]
    recording_dir = os.path.basename(os.path.dirname(video_path))
    per_roi_rows: dict[str, list[dict]] = {roi['name']: [] for roi in rois}

    # ── Pre-allocate expected frame rows ──────────────────────────────
    frames_data: list[dict] = []
    effective_fps = float(fps) if fps and fps > 0 else 30.0
    duration = (total_frames / effective_fps) if effective_fps > 0 else 0.0

    if save_tracked_video:
        n_expected = max(total_frames, 1)
        dt = 1.0 / effective_fps
        frames_data = [
            {"timestamp": round(i * dt, 6), "roi_data": None}
            for i in range(n_expected)
        ]

    # Process each frame
    frame_idx = 0

    try:
        if isinstance(frame_buffer, list) and len(frame_buffer) > 0:
            frame_iter = iter(frame_buffer)
            use_cap_read = False
        else:
            frame_iter = None
            use_cap_read = True

        first_inference_error_logged = False
        while True:
            if use_cap_read:
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                except Exception:
                    timestamp = (frame_idx / fps) if fps and fps > 0 else float('nan')
            else:
                try:
                    item = next(frame_iter)
                except StopIteration:
                    break
                try:
                    frame, timestamp = item
                except Exception:
                    frame, timestamp = None, None
                if frame is None:
                    break

            # ── Batch inference ───────────────────────────────────────
            roi_images = []
            for roi in rois:
                x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
                roi_img = frame[y:y+h, x:x+w]
                if len(roi_img.shape) == 3:
                    roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
                roi_images.append(roi_img)

            try:
                all_coords = infer_fn(roi_images)
            except InterruptedError:
                raise
            except Exception as e:
                if not first_inference_error_logged:
                    import traceback as _tb
                    print(f"[tracker] Batch inference failed: {e}")
                    _tb.print_exc()
                    first_inference_error_logged = True
                all_coords = [None] * len(rois)

            for i, roi in enumerate(rois):
                coords = all_coords[i] if i < len(all_coords) else None
                if coords is not None:
                    coords = np.asarray(coords).squeeze()
                    if coords.ndim == 2 and coords.shape[0] >= 2:
                        length = float(np.linalg.norm(coords[1] - coords[0]))
                    else:
                        length = float('nan')
                else:
                    length = float('nan')
                    coords = None

                if save_inferences and coords is not None \
                        and coords.ndim == 2 and coords.shape[0] >= 2 \
                        and frame_idx % 30 == 0:
                    try:
                        from functions.inference_collector import save_sample
                        inferences_dir = os.path.join(output_folder, "inferences")
                        save_sample(
                            roi_gray=roi_images[i],
                            coords=coords,
                            model_name=model_name or "unknown",
                            target_size=target_size,
                            recording_dir=recording_dir,
                            recording=recording_name,
                            roi_name=roi['name'],
                            frame_idx=frame_idx,
                            data_dir=inferences_dir,
                        )
                    except Exception as _ce:
                        print(f"[tracker] Failed to save inference sample: {_ce}")

                try:
                    t_val = float(timestamp) if timestamp is not None else float('nan')
                except Exception:
                    t_val = float('nan')

                per_roi_rows[roi['name']].append({'time': t_val, 'length': float(length)})

            # ── Fill pre-allocated frame row ──────────────────────────
            if save_tracked_video and frames_data:
                try:
                    ts = float(timestamp) if timestamp is not None else float('nan')
                except (ValueError, TypeError):
                    ts = float('nan')
                if ts >= 0:
                    target_idx = int(round(ts * effective_fps))
                else:
                    target_idx = frame_idx
                target_idx = max(0, min(target_idx, len(frames_data) - 1))

                if frames_data[target_idx]["roi_data"] is None:
                    roi_entries = []
                    for i, roi in enumerate(rois):
                        ci = all_coords[i] if i < len(all_coords) else None
                        coords_list = None
                        if ci is not None:
                            ci_arr = np.asarray(ci).squeeze()
                            if ci_arr.ndim == 2 and ci_arr.shape[0] >= 2:
                                coords_list = ci_arr.tolist()
                        _, png_bytes = cv2.imencode('.png', roi_images[i])
                        img_b64 = base64.b64encode(png_bytes.tobytes()).decode('ascii')
                        roi_entries.append({
                            'name': roi['name'],
                            'image_b64': img_b64,
                            'coords': coords_list,
                        })
                    frames_data[target_idx]["roi_data"] = roi_entries

            if progress_callback:
                try:
                    cur_t = float(timestamp) if timestamp is not None else float('nan')
                except (ValueError, TypeError):
                    cur_t = float('nan')
                if not (cur_t >= 0):
                    cur_t = (frame_idx / fps) if fps and fps > 0 else float(frame_idx)
                progress_callback(cur_t, duration, f"Processing {cur_t:.1f}s / {duration:.1f}s")

            frame_idx += 1

        if progress_callback:
            progress_callback(duration, duration, "Saving results...")

        # ── Write tracked-frames JSON ─────────────────────────────────
        tracked_frames_json = None
        if save_tracked_video:
            json_path = os.path.join(output_folder, f'.{recording_name}_tracked_frames.json')
            metadata = {
                'recording_name': recording_name,
                'fps': effective_fps,
                'total_frames': total_frames,
                'duration': float(duration),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'rois': [{'name': r['name'], 'width': r['width'], 'height': r['height']}
                          for r in rois],
                'frames': frames_data,
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f)
            tracked_frames_json = json_path
            filled = sum(1 for fr in frames_data if fr["roi_data"] is not None)
            print(f"[tracker] Saved tracked frames data: {json_path} "
                  f"({filled}/{len(frames_data)} frames filled, {len(rois)} ROIs)")

        # ── Save per-ROI length CSVs ──────────────────────────────────
        saved_files = []
        for roi in rois:
            roi_name = roi['name']
            rows = per_roi_rows.get(roi_name, [])
            if not rows:
                continue
            df_raw = pd.DataFrame(rows, columns=['time', 'length'])
            t = df_raw['time'].to_numpy(dtype=float)
            l_vals = df_raw['length'].to_numpy(dtype=float)
            valid = np.isfinite(t) & np.isfinite(l_vals)
            t_clean = t[valid]
            l_clean = l_vals[valid]
            if len(t_clean) < 2:
                continue

            diffs = np.diff(t_clean)
            dt = float(np.min(diffs[diffs > 0])) if np.any(diffs > 0) else 0.0
            if dt <= 0:
                df_out = df_raw
            else:
                t_start, t_end = t_clean[0], t_clean[-1]
                n_steps = int(round((t_end - t_start) / dt))
                t_uniform = t_start + dt * np.arange(n_steps + 1, dtype=float)
                t_uniform[-1] = t_end
                l_interp = np.interp(t_uniform, t_clean, l_clean)
                df_out = pd.DataFrame({'time': t_uniform, 'length': l_interp})

            aid = roi.get('sample_id', roi_name)
            rec = roi.get('recording_name', recording_name)
            filename = f"{aid}_{rec}_{roi_name}_length.csv"
            rel_path = f"lengths/{filename}"
            csv_path = os.path.join(output_folder, rel_path)
            df_out.to_csv(csv_path, index=False)
            saved_files.append(rel_path)
            print(f"Saved ROI length CSV: {csv_path} ({len(df_out)} rows, dt={dt:.6f}s)")

        result = {
            'success': True,
            'message': f"Processed {total_frames} frames for {len(rois)} ROI(s). Saved {len(saved_files)} CSV(s).",
            'saved_files': saved_files,
        }
        if tracked_frames_json:
            result['tracked_frames_json'] = tracked_frames_json
        return result

    except InterruptedError:
        raise
    except Exception as e:
        import traceback
        error_msg = f"Error during tracking: {str(e)}\n{traceback.format_exc()}"
        return {'success': False, 'message': error_msg}

    finally:
        cap.release()


# ── Tracked video composition (post-tracking) ───────────────────────────────

def compose_tracked_video_from_json(json_path: str) -> str:
    """Build a frame-aligned tracked video from a tracking-frames JSON file.

    Layout (top → bottom, inside 16 px border)::

        ┌──────────────────────────────────────────┐
        │     Recording Name  (16 px bold)         │  TITLE_HEIGHT = 16
        ├──────────────────────────────────────────┤  SEP_HEIGHT = 8 (black)
        │  ROI 1   │  ROI 2   │  ROI 3  …         │  ROI_DISPLAY_SIZE = 256
        │ (aspect- │ (aspect- │                     │  (larger dim → 256,
        │  ratio   │  ratio   │                     │   other proportional)
        │  kept)   │  kept)   │                     │
        ├──────────────────────────────────────────┤  SEP_HEIGHT = 8 (black)
        │   ROI 1  │   ROI 2  │   ROI 3  …         │  LABEL_HEIGHT = 14
        └──────────────────────────────────────────┘

    Each expected frame row is scanned in order.  Rows whose ``roi_data`` is
    ``null`` (dropped frames) repeat the most recent valid row (forward-fill).

    Parameters
    ----------
    json_path : str
        Path to the ``.{recording}_tracked_frames.json`` temp file.

    Returns
    -------
    str
        Relative path to the generated tracked video
        (``tracked_video/<recording>_tracked.mp4``).
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    recording_name = data['recording_name']
    out_fps = data['fps']
    total_frames = data['total_frames']
    rois_meta = data['rois']
    frames_data = data['frames']
    num_rois = len(rois_meta)

    canvas_w = _build_canvas_width(num_rois)
    canvas_h = CANVAS_HEIGHT

    output_dir = os.path.dirname(json_path)
    video_dir = os.path.join(output_dir, "tracked_video")
    os.makedirs(video_dir, exist_ok=True)
    out_path = os.path.join(video_dir, f"{recording_name}_tracked.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(out_path, fourcc, out_fps, (canvas_w, canvas_h))
    if not video_writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    if not frames_data:
        video_writer.release()
        raise ValueError("No frames in tracking data")

    # Pre-compute ROI column x-positions (inner coords, relative to content area)
    roi_col_x = [ROI_GAP + i * (ROI_DISPLAY_SIZE + ROI_GAP) for i in range(num_rois)]

    # y-bands (inner coords, relative to content area; BORDER added at draw time)
    y_title = 0
    y_sep1 = TITLE_HEIGHT
    y_roi = TITLE_HEIGHT + SEP_HEIGHT
    y_sep2 = y_roi + ROI_DISPLAY_SIZE
    y_label = y_sep2 + SEP_HEIGHT

    last_valid_roi_data = None  # forward-fill state

    for fi, frame_entry in enumerate(frames_data):
        roi_data = frame_entry.get("roi_data")
        if roi_data is not None:
            last_valid_roi_data = roi_data

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # ── Title: recording name (16 px bold), centred ────────────────
        title_text = recording_name
        (tw, th), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        title_x = BORDER + (canvas_w - 2 * BORDER - tw) // 2
        title_y = BORDER + y_title + (TITLE_HEIGHT + th) // 2
        cv2.putText(canvas, title_text, (title_x, title_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # ── Separator 1 (pure black, 8 px) ─────────────────────────────
        canvas[BORDER + y_sep1:BORDER + y_sep1 + SEP_HEIGHT,
               BORDER:BORDER + (canvas_w - 2 * BORDER)] = (0, 0, 0)

        # ── ROI row ────────────────────────────────────────────────────
        source = last_valid_roi_data if last_valid_roi_data is not None else []
        for i, roi_meta in enumerate(rois_meta):
            roi_name = roi_meta['name']
            orig_w = roi_meta['width']
            orig_h = roi_meta['height']
            cx = BORDER + roi_col_x[i]   # absolute x on full canvas

            # Find matching entry in source
            entry = None
            for rd in source:
                if rd['name'] == roi_name:
                    entry = rd
                    break

            if entry and entry.get('image_b64'):
                img_bytes = base64.b64decode(entry['image_b64'])
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                roi_img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                if roi_img is not None:
                    # Scale to fit 256×256 preserving aspect ratio
                    # (larger dimension → 256, other scaled proportionally)
                    scale = ROI_DISPLAY_SIZE / max(orig_w, orig_h)
                    disp_w = int(orig_w * scale)
                    disp_h = int(orig_h * scale)

                    roi_resized = cv2.resize(roi_img, (disp_w, disp_h),
                                             interpolation=cv2.INTER_LINEAR)
                    roi_bgr = cv2.cvtColor(roi_resized, cv2.COLOR_GRAY2BGR)

                    # Centre within the 256×256 cell
                    ox = (ROI_DISPLAY_SIZE - disp_w) // 2
                    oy = (ROI_DISPLAY_SIZE - disp_h) // 2

                    ry = BORDER + y_roi
                    canvas[ry + oy:ry + oy + disp_h,
                           cx + ox:cx + ox + disp_w] = roi_bgr

                    # Draw red crosshairs at pillar coords (scaled + offset)
                    coords = entry.get('coords')
                    if coords and len(coords) >= 2:
                        try:
                            for px, py in coords:
                                gx = cx + ox + int(px * scale)
                                gy = ry + oy + int(py * scale)
                                cv2.line(canvas, (gx - 5, gy), (gx + 5, gy),
                                         (0, 0, 255), 1)
                                cv2.line(canvas, (gx, gy - 5), (gx, gy + 5),
                                         (0, 0, 255), 1)
                        except (ValueError, OverflowError):
                            pass

            # ── ROI name label (14 px, centred under cell) ──────────────
            label = roi_name
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            label_x = cx + (ROI_DISPLAY_SIZE - lw) // 2
            label_y = BORDER + y_label + lh
            cv2.putText(canvas, label, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Separator 2 (pure black, 8 px) ─────────────────────────────
        canvas[BORDER + y_sep2:BORDER + y_sep2 + SEP_HEIGHT,
               BORDER:BORDER + (canvas_w - 2 * BORDER)] = (0, 0, 0)

        # ── Four-side border (16 px, pure black) ───────────────────────
        # Top edge
        canvas[0:BORDER, :] = (0, 0, 0)
        # Bottom edge
        canvas[canvas_h - BORDER:canvas_h, :] = (0, 0, 0)
        # Left edge
        canvas[:, 0:BORDER] = (0, 0, 0)
        # Right edge
        canvas[:, canvas_w - BORDER:canvas_w] = (0, 0, 0)

        video_writer.write(canvas)

    video_writer.release()
    rel_path = f"tracked_video/{recording_name}_tracked.mp4"
    print(f"[tracker] Tracked video saved: {out_path} "
          f"({total_frames} frames, {canvas_w}x{canvas_h})")
    return rel_path
