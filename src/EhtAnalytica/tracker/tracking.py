"""
Tracking module for processing video frames with ROI-based tracking.
"""

import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path


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
    
    # Prepare output writers for each ROI
    roi_data = {}
    for roi in rois:
        roi_name = roi['name']
        roi_width = roi['width']
        roi_height = roi['height']
        
        # Create video writer for this ROI with '_tracked' suffix
        output_video_path = os.path.join(output_folder, f"{roi_name}_tracked.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (roi_width, roi_height))
        
        roi_data[roi_name] = {
            'roi': roi,
            'writer': writer,
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
                # tracker_model.predict() should return 2 sets of (x, y) coordinates
                coords = tracker_model.predict(roi_image)
                
                # coords should be shape (2, 2): [[x1, y1], [x2, y2]]
                if coords is not None and len(coords) == 2:
                    x1, y1 = coords[0]
                    x2, y2 = coords[1]
                    
                    # Draw 'X' marks at the two coordinates
                    # First coordinate - Green X
                    draw_x_marker(roi_image, int(x1), int(y1), (0, 255, 0), size=10, thickness=2)
                    
                    # Second coordinate - Red X
                    draw_x_marker(roi_image, int(x2), int(y2), (0, 0, 255), size=10, thickness=2)
                
                # Write frame to output video (even if no valid prediction)
                data['writer'].write(roi_image)
            
            # Update progress every 10 frames
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, total_frames, f"Processing frame {frame_idx}/{total_frames}")
            
            frame_idx += 1
        
        # Final progress update
        if progress_callback:
            progress_callback(total_frames, total_frames, "Saving results...")
        
        # Release video writers
        for roi_name, data in roi_data.items():
            # Release video writer
            data['writer'].release()
            
            print(f"Saved {roi_name}: {data['video_path']}")
        
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
