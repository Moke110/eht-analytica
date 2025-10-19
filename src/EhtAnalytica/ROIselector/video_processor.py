import cv2
import numpy as np
import os
from pathlib import Path
import threading


class VideoProcessor:
    """Process and validate video recordings for EHT Analytica."""
    
    def __init__(self):
        self.video_path = None
        self.metadata = {}
        self.valid_frames = []
        self.frame_timestamps = []
        
    def get_video_metadata(self, video_path):
        """
        Get metadata from video file.
        
        Parameters
        ----------
        video_path : str
            Path to the video file
            
        Returns
        -------
        dict
            Dictionary containing video metadata
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        metadata = {
            'path': video_path,
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
        }
        
        cap.release()
        return metadata
    
    def scan_video_frames(self, video_path, progress_callback=None):
        """
        Scan through video to get all readable frames and timestamps.
        
        Parameters
        ----------
        video_path : str
            Path to the video file
        progress_callback : callable, optional
            Callback function to report progress
            
        Returns
        -------
        tuple
            (list of frame indices, list of timestamps in ms)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        valid_frames = []
        timestamps = []
        frame_idx = 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame is not None and frame.size > 0:
                valid_frames.append(frame_idx)
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            
            frame_idx += 1
            
            if progress_callback and frame_idx % 10 == 0:
                progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                progress_callback(progress)
        
        cap.release()
        return valid_frames, timestamps
    
    def fix_video_frames(self, video_path, metadata, valid_frames, output_path=None):
        """
        Fix video by filling missing frames if scan result doesn't match metadata.
        
        Parameters
        ----------
        video_path : str
            Path to the original video file
        metadata : dict
            Video metadata
        valid_frames : list
            List of valid frame indices from scan
        output_path : str, optional
            Path for output video (if None, overwrites original)
            
        Returns
        -------
        str
            Path to the fixed video file
        """
        expected_frames = metadata['frame_count']
        actual_frames = len(valid_frames)
        
        # If frames match, no fix needed
        if expected_frames == actual_frames:
            return video_path
        
        # Prepare output path
        if output_path is None:
            path_obj = Path(video_path)
            temp_path = path_obj.parent / f"{path_obj.stem}_fixed{path_obj.suffix}"
            output_path = str(temp_path)
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = metadata['fps']
        width = metadata['width']
        height = metadata['height']
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        last_valid_frame = None
        
        for expected_idx in range(expected_frames):
            if expected_idx in valid_frames:
                ret, frame = cap.read()
                if ret and frame is not None:
                    out.write(frame)
                    last_valid_frame = frame
                    frame_idx += 1
            else:
                # Fill gap with last valid frame or black frame
                if last_valid_frame is not None:
                    out.write(last_valid_frame)
                else:
                    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    out.write(black_frame)
        
        cap.release()
        out.release()
        
        # Delete original and rename fixed video to original name
        if output_path.endswith('_fixed.mp4') or output_path.endswith(f'_fixed{Path(video_path).suffix}'):
            try:
                os.remove(video_path)  # Delete original
                os.rename(output_path, video_path)  # Rename fixed to original
                return video_path
            except Exception as e:
                print(f"Warning: Could not replace original file: {e}")
                return output_path
        
        return output_path
    
    def process_video(self, video_path, progress_callback=None, completion_callback=None):
        """
        Complete video processing pipeline: get metadata, scan frames, fix if needed.
        
        Parameters
        ----------
        video_path : str
            Path to the video file
        progress_callback : callable, optional
            Callback to report progress
        completion_callback : callable, optional
            Callback when processing is complete with (metadata, first_frame)
        """
        try:
            # Step 1: Get metadata
            if progress_callback:
                progress_callback(0, "Getting video metadata...")
            
            metadata = self.get_video_metadata(video_path)
            self.metadata = metadata
            
            # Step 2: Scan frames
            if progress_callback:
                progress_callback(10, "Scanning video frames...")
            
            def scan_progress(percent):
                if progress_callback:
                    progress_callback(10 + percent * 0.7, "Scanning video frames...")
            
            valid_frames, timestamps = self.scan_video_frames(video_path, scan_progress)
            self.valid_frames = valid_frames
            self.frame_timestamps = timestamps
            
            # Step 3: Fix video if needed
            if len(valid_frames) != metadata['frame_count']:
                if progress_callback:
                    progress_callback(80, "Fixing video frames...")
                
                video_path = self.fix_video_frames(video_path, metadata, valid_frames)
                
                # Re-scan to update metadata
                metadata = self.get_video_metadata(video_path)
                self.metadata = metadata
            
            # Get first frame
            if progress_callback:
                progress_callback(95, "Loading first frame...")
            
            cap = cv2.VideoCapture(video_path)
            ret, first_frame = cap.read()
            cap.release()
            
            if not ret:
                raise ValueError("Could not read first frame")
            
            if progress_callback:
                progress_callback(100, "Complete!")
            
            # Call completion callback
            if completion_callback:
                completion_callback(metadata, first_frame)
        
        except Exception as e:
            if completion_callback:
                completion_callback(None, None, error=str(e))


def process_video_threaded(video_path, progress_callback=None, completion_callback=None):
    """
    Process video in a separate thread to prevent GUI freezing.
    
    Parameters
    ----------
    video_path : str
        Path to the video file
    progress_callback : callable, optional
        Callback to report progress (percent, message)
    completion_callback : callable, optional
        Callback when complete (metadata, first_frame, error=None)
        
    Returns
    -------
    threading.Thread
        The processing thread
    """
    processor = VideoProcessor()
    
    def worker():
        processor.process_video(video_path, progress_callback, completion_callback)
    
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    
    return thread
