"""
Tracking module for processing video frames with ROI-based tracking.
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


def get_fourcc(code='mp4v'):
    """
    Robust helper to return a fourcc code integer for VideoWriter.
    Some OpenCV builds or type stubs may not expose VideoWriter_fourcc;
    this helper tries common attributes and falls back to 0.
    """
    try:
        vfunc = getattr(cv2, 'VideoWriter_fourcc', None)
        if vfunc is not None:
            try:
                return vfunc(*code)
            except Exception:
                pass
        # Older OpenCV might have FOURCC
        vfunc = getattr(cv2, 'FOURCC', None)
        if vfunc is not None:
            try:
                return vfunc(*code)
            except Exception:
                pass
    except Exception:
        pass
    return 0


def draw_x_marker(image, x, y, color, size=10, thickness=2):
    """
    Draw an 'X' marker at the specified position.
    
    Parameters
    ----------
    image : np.ndarray
        Image to draw on
    x : int
        X coordinate
    y : int
        Y coordinate
    color : tuple
        BGR color tuple
    size : int
        Size of the X marker (half the diagonal length)
    thickness : int
        Line thickness
    """
    # Draw two diagonal lines to form an X
    cv2.line(image, (x - size, y - size), (x + size, y + size), color, thickness)
    cv2.line(image, (x - size, y + size), (x + size, y - size), color, thickness)


def process_tracking(video_path, rois, tracker_model, output_folder, progress_callback=None, total_frames_hint: int | None = None, frame_buffer: list | None = None):
    """
    Process video tracking for each ROI.
    
    Parameters
    ----------
    video_path : str
        Path to the input video file
    rois : list of dict
        List of ROI dictionaries with keys: 'name', 'x', 'y', 'width', 'height', 'color'
    tracker_model : TrackerModel
        Loaded tracker model instance with predict() method
    output_folder : str
        Path to the output folder for saving videos
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
    
    # Prepare output writers and dataframes for each ROI
    roi_data = {}
    for roi in rois:
        roi_name = roi['name']
        roi_width = roi['width']
        roi_height = roi['height']

        # Create video writer for this ROI with '_tracked' suffix
        output_video_path = os.path.join(output_folder, f"{roi_name}_tracked.mp4")
        # Use robust fourcc helper (handles OpenCV builds where VideoWriter_fourcc isn't exported to stubs)
        fourcc = get_fourcc('mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (roi_width, roi_height))

        # Create dataframe for this ROI (only timestamp and length)
        df = pd.DataFrame(columns=['timestamp', 'length'])

        roi_data[roi_name] = {
            'roi': roi,
            'writer': writer,
            'dataframe': df,
            'video_path': output_video_path
        }
    
    # Process each frame
    frame_idx = 0

    try:
        # If a preprocessed buffer is provided, use it (each entry: (frame_bgr, timestamp_seconds))
        if isinstance(frame_buffer, list) and len(frame_buffer) > 0:
            # Create an explicit iterator to satisfy static checkers
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
                # Use real timestamp reported by capture in seconds
                try:
                    timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                except Exception:
                    # Fallback to frame index based timestamp if CAP_PROP_POS_MSEC unsupported
                    timestamp = (frame_idx / fps) if fps and fps > 0 else float('nan')
            else:
                try:
                    item = next(frame_iter)  # type: ignore[arg-type]
                except StopIteration:
                    break
                # Expect item as (frame, timestamp_seconds)
                try:
                    frame, timestamp = item
                except Exception:
                    # If shape doesn't match, skip
                    frame, timestamp = None, None
                if frame is None:
                    break
            
            # Process each ROI
            for roi_name, data in roi_data.items():
                roi = data['roi']
                x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
                
                # Extract ROI from original frame
                roi_image = frame[y:y+h, x:x+w].copy()
                
                try:
                    coords = tracker_model.forward(roi_image)
                except Exception:
                    coords = None

                # Fallback for TorchScript models: call the scripted model with a
                # Fallback for TorchScript models: build grayscale torch.Tensor [H, W]
                # and convert output to numpy. Robustly detect the model device
                # (from parameters/buffers) and move the tensor there.
                if coords is None and torch is not None:
                    try:
                        if len(roi_image.shape) == 3:
                            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
                        else:
                            gray = roi_image

                        tensor = torch.from_numpy(gray).float()

                        # Local alias to help static analysis
                        torch_local = torch

                        # Robust device detection for scripted or nn.Module models
                        def _get_model_device(m):
                            # Try attr first
                            try:
                                d = getattr(m, 'device', None)
                                if d is not None:
                                    return d
                            except Exception:
                                pass
                            # Try parameters
                            try:
                                for p in m.parameters():
                                    return p.device
                            except Exception:
                                pass
                            # Try buffers
                            try:
                                for b in m.buffers():
                                    return b.device
                            except Exception:
                                pass
                            # Fallback
                            # Prefer CUDA if available, otherwise CPU. Guard against
                            # torch being None (though this branch is only entered
                            # when torch is available).
                            if torch_local is not None and hasattr(torch_local, 'cuda') and torch_local.cuda.is_available():
                                return torch_local.device('cuda')
                            return torch_local.device('cpu') if torch_local is not None else None

                        model_device = _get_model_device(tracker_model)

                        # Move tensor to model device (no-op if already same)
                        tensor = tensor.to(model_device)

                        with torch.no_grad():
                            # The scripted tracker expects a 2D [H, W] tensor.
                            # Some older saved modules may expect [1,1,H,W]; try both.
                            out = None
                            try:
                                out = tracker_model(tensor)
                            except Exception:
                                t2 = tensor.unsqueeze(0).unsqueeze(0)
                                out = tracker_model(t2)

                        # Unwrap output if needed and convert to numpy
                        if isinstance(out, (list, tuple)):
                            out = out[0]

                        if isinstance(out, np.ndarray):
                            coords = out
                        else:
                            try:
                                coords = out.detach().cpu().numpy()
                            except Exception:
                                try:
                                    coords = np.array(out)
                                except Exception:
                                    coords = None
                    except Exception:
                        coords = None
                
                # coords should be shape (2, 2): [[x1, y1], [x2, y2]]
                if coords is not None and len(coords) == 2:
                    x1, y1 = coords[0]
                    x2, y2 = coords[1]
                    
                    # Calculate pixel distance (length)
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    # Draw 'X' marks at the two coordinates
                    # First coordinate - Green X
                    draw_x_marker(roi_image, int(x1), int(y1), (0, 255, 0), size=10, thickness=2)
                    
                    # Second coordinate - Red X
                    draw_x_marker(roi_image, int(x2), int(y2), (0, 0, 255), size=10, thickness=2)
                    
                    # Add to dataframe (only timestamp and length)
                    new_row = pd.DataFrame([
                        {
                            'timestamp': timestamp,
                            'length': length
                        }
                    ])
                    data['dataframe'] = pd.concat([data['dataframe'], new_row], ignore_index=True)
                else:
                    # No valid prediction, add NaN for length
                    new_row = pd.DataFrame([
                        {
                            'timestamp': timestamp,
                            'length': np.nan
                        }
                    ])
                    data['dataframe'] = pd.concat([data['dataframe'], new_row], ignore_index=True)
                
                # Write frame to output video
                data['writer'].write(roi_image)
            
            # Update progress every 10 frames
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, progress_total, f"Processing frame {frame_idx}/{progress_total}")
            
            frame_idx += 1
        
        # Final progress update
        if progress_callback:
            progress_callback(progress_total, progress_total, "Saving results...")
        
        # Release video writers and save dataframes
        for roi_name, data in roi_data.items():
            # Release video writer
            data['writer'].release()
            
            # Save dataframe as CSV (only timestamp and length)
            csv_path = os.path.join(output_folder, f"{roi_name}_tracked.csv")
            data['dataframe'].to_csv(csv_path, index=False)
            
            print(f"Saved {roi_name}: {data['video_path']} and {csv_path}")
        
        return {
            'success': True, 
            'message': f"Successfully processed {total_frames} frames for {len(rois)} ROI(s)."
        }
        
    except Exception as e:
        import traceback
        error_msg = f"Error during tracking: {str(e)}\n{traceback.format_exc()}"
        return {'success': False, 'message': error_msg}
        
    finally:
        # Clean up
        cap.release()
        for data in roi_data.values():
            if data['writer'].isOpened():
                data['writer'].release()
