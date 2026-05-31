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
        # Buffer of tuples: (frame ndarray (BGR), timestamp_seconds: float)
        self.frame_buffer = []
        
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
        timestamps = []  # in ms
        frames_buffer = []  # list of (frame, timestamp_seconds)
        frame_idx = 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame is not None and frame.size > 0:
                valid_frames.append(frame_idx)
                ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamps.append(ts_ms)
                # store frame copy with timestamp in seconds
                try:
                    frames_buffer.append((frame.copy(), float(ts_ms) / 1000.0))
                except Exception:
                    frames_buffer.append((frame, float(ts_ms) / 1000.0))
            
            frame_idx += 1
            
            if progress_callback and frame_idx % 10 == 0:
                progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                progress_callback(progress)
        
        cap.release()
        # cache buffer for downstream use
        self.frame_buffer = frames_buffer
        return valid_frames, timestamps
    
    # Removed video fixing; processing only scans frames and timestamps in original video.
    
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
            
            # Step 3: No fixing; keep original video. Continue to first frame.
            
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
            
            # Call completion callback (pass frame buffer and valid frame count)
            if completion_callback:
                completion_callback(metadata, first_frame, None, self.frame_buffer, len(valid_frames))
        
        except Exception as e:
            if completion_callback:
                completion_callback(None, None, error=str(e), frame_buffer=None, valid_count=0)


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
        Callback when complete (metadata, first_frame, error=None, frame_buffer=list, valid_count=int)
        
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
