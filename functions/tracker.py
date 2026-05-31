"""
Tracking module for processing video frames with ROI-based tracking.

This implementation generates a single time-length CSV per recording without
creating annotated frames or tracked videos.
"""

import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path

# Optional torch import (used only when tracker is a TorchScript module)
try:
    import torch
except Exception:
    torch = None


def process_tracking(video_path, rois, tracker_model, output_folder, progress_callback=None, total_frames_hint: int | None = None, frame_buffer: list | None = None):
    """
    Process video tracking for each ROI and save a single time-length CSV.
    
    Parameters
    ----------
    video_path : str
        Path to the input video file
    rois : list of dict
        List of ROI dictionaries with keys: 'name', 'x', 'y', 'width', 'height', 'color'
    tracker_model : TrackerModel
        Loaded tracker model instance with predict() method
    output_folder : str
        Path to the output folder for saving outputs (CSV)
    progress_callback : callable, optional
        Callback function(current_frame, total_frames, message) for progress updates
    
    Returns
    -------
    dict
        Dictionary with 'success': bool and 'message': str
    """
    
    if not rois:
        return {'success': False, 'message': 'No ROIs defined. Please draw ROIs first.'}
    
    # Open video to query properties (fps); when frame_buffer is provided, we
    # won't read frames from cap, only use metadata.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'success': False, 'message': f'Could not open video: {video_path}'}

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # If using preprocessed buffer of valid frames, progress should reflect that length
    if isinstance(frame_buffer, list) and len(frame_buffer) > 0:
        progress_total = len(frame_buffer)
    else:
        progress_total = total_frames_hint if (isinstance(total_frames_hint, int) and total_frames_hint > 0) else total_frames
    
    # Resolve model device once before the loop
    model_device = None
    if torch is not None:
        try:
            d = getattr(tracker_model, 'device', None)
            if d is not None:
                model_device = d
        except Exception:
            pass
        if model_device is None:
            try:
                for p in tracker_model.parameters():
                    model_device = p.device
                    break
            except Exception:
                pass
        if model_device is None:
            try:
                for b in tracker_model.buffers():
                    model_device = b.device
                    break
            except Exception:
                pass
        if model_device is None:
            model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare per-ROI CSV builders
    os.makedirs(output_folder, exist_ok=True)
    per_roi_rows: dict[str, list[dict]] = {roi['name']: [] for roi in rois}

    # Process each frame
    frame_idx = 0

    try:
        # If a preprocessed buffer is provided, use it (each entry: (frame_bgr, timestamp_seconds))
        if isinstance(frame_buffer, list) and len(frame_buffer) > 0:
            frame_iter = iter(frame_buffer)
            use_cap_read = False
        else:
            frame_iter = None
            use_cap_read = True

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

            # Process each ROI and append individual rows
            for roi in rois:
                x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
                roi_image = frame[y:y+h, x:x+w]

                coords = None
                if torch is not None and model_device is not None:
                    try:
                        if len(roi_image.shape) == 3:
                            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
                        else:
                            gray = roi_image

                        tensor = torch.from_numpy(gray.copy()).float().to(model_device)

                        with torch.inference_mode():
                            out = tracker_model(tensor)

                        if isinstance(out, (list, tuple)):
                            out = out[0]
                        if isinstance(out, np.ndarray):
                            coords = out
                        else:
                            coords = out.detach().cpu().numpy()
                    except Exception:
                        coords = None

                if coords is not None and hasattr(coords, '__len__') and len(coords) == 2:
                    x1, y1 = coords[0]
                    x2, y2 = coords[1]
                    length = float(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
                else:
                    length = float('nan')

                try:
                    t_val = float(timestamp) if timestamp is not None else float('nan')
                except Exception:
                    t_val = float('nan')

                per_roi_rows[roi['name']].append({'time': t_val, 'length': float(length)})

            if progress_callback:
                progress_callback(frame_idx, progress_total, f"Processing frame {frame_idx}/{progress_total}")

            frame_idx += 1
        
        if progress_callback:
            progress_callback(progress_total, progress_total, "Saving results...")

        # Save per-ROI time-length CSVs: {roi_name}_length.csv
        saved_files = []
        for roi_name, rows in per_roi_rows.items():
            df_out = pd.DataFrame(rows, columns=['time', 'length'])
            csv_path = os.path.join(output_folder, f"{roi_name}_length.csv")
            df_out.to_csv(csv_path, index=False)
            saved_files.append(os.path.basename(csv_path))
            print(f"Saved ROI time-length CSV: {csv_path}")

        return {
            'success': True, 
            'message': f"Processed {total_frames} frames for {len(rois)} ROI(s). Saved {len(saved_files)} CSV(s)."
        }
        
    except Exception as e:
        import traceback
        error_msg = f"Error during tracking: {str(e)}\n{traceback.format_exc()}"
        return {'success': False, 'message': error_msg}
        
    finally:
        cap.release()
