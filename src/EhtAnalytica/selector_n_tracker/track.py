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


def process_tracking(video_path, rois, tracker_model, output_folder, progress_callback=None):
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
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'success': False, 'message': f'Could not open video: {video_path}'}
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
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
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate timestamp (in seconds)
            timestamp = frame_idx / fps
            
            # Process each ROI
            for roi_name, data in roi_data.items():
                roi = data['roi']
                x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
                
                # Extract ROI from original frame
                roi_image = frame[y:y+h, x:x+w].copy()
                
                # Get predictions from tracker model
                coords = None

                # Preferred API: tracker_model.predict(image) (existing Python wrapper)
                try:
                    coords = tracker_model.forward(roi_image)
                except Exception:
                    coords = None

                # Fallback for TorchScript models: call the scripted model with a
                # grayscale torch.Tensor [H, W] and convert output to numpy
                if coords is None and torch is not None:
                    try:
                        if len(roi_image.shape) == 3:
                            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
                        else:
                            gray = roi_image

                        tensor = torch.from_numpy(gray).float()
                        with torch.no_grad():
                            out = tracker_model(tensor)

                        if isinstance(out, (list, tuple)):
                            out = out[0]

                        # try to convert to numpy
                        if isinstance(out, np.ndarray):
                            coords = out
                        else:
                            try:
                                coords = out.detach().cpu().numpy()
                            except Exception:
                                coords = np.array(out)
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
                    new_row = pd.DataFrame([{
                        'timestamp': timestamp,
                        'length': length
                    }])
                    data['dataframe'] = pd.concat([data['dataframe'], new_row], ignore_index=True)
                else:
                    # No valid prediction, add NaN for length
                    new_row = pd.DataFrame([{
                        'timestamp': timestamp,
                        'length': np.nan
                    }])
                    data['dataframe'] = pd.concat([data['dataframe'], new_row], ignore_index=True)
                
                # Write frame to output video
                data['writer'].write(roi_image)
            
            # Update progress every 10 frames
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, total_frames, f"Processing frame {frame_idx}/{total_frames}")
            
            frame_idx += 1
        
        # Final progress update
        if progress_callback:
            progress_callback(total_frames, total_frames, "Saving results...")
        
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
