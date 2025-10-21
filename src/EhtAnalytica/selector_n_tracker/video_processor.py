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
    
    def fix_video_frames(self, video_path, metadata, valid_frames, timestamps=None, output_path=None):
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

        # Create video writer (robust getattr to avoid static analysis issues)
        vfunc = getattr(cv2, 'VideoWriter_fourcc', None)
        if vfunc is not None:
            try:
                fourcc = vfunc(*'mp4v')
            except Exception:
                fourcc = 0
        else:
            fourcc = 0
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # If timestamps weren't provided, try to use stored frame_timestamps
        if timestamps is None:
            timestamps = getattr(self, 'frame_timestamps', None)

        # If we have readable timestamps, build mapping from target frames -> nearest readable frame
        if timestamps and len(timestamps) > 0 and fps and fps > 0:
            # timestamps are in ms
            time_per_frame = 1000.0 / fps
            # Build list of readable frame timestamps (aligned with valid_frames)
            readable_timestamps = list(timestamps)
            readable_frames = list(valid_frames)

            # Preload readable frames into memory to avoid repeated seeks
            loaded_frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            read_map = {fidx: None for fidx in readable_frames}
            next_read_idx = 0
            total_readable = len(readable_frames)
            # Read sequentially and save frames whose indices are in readable_frames
            current_frame_idx = 0
            while next_read_idx < total_readable:
                ret, frame = cap.read()
                if not ret:
                    break
                if current_frame_idx == readable_frames[next_read_idx]:
                    loaded_frames.append(frame.copy())
                    next_read_idx += 1
                current_frame_idx += 1

            # If we couldn't preload correctly (counts mismatch), fallback to reading by timestamp
            if len(loaded_frames) != total_readable:
                # Rebuild loaded_frames by seeking to each readable timestamp
                loaded_frames = []
                for ts in readable_timestamps:
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        loaded_frames.append(frame.copy())
                    else:
                        # if seek/read fails, append a black frame
                        loaded_frames.append(np.zeros((height, width, 3), dtype=np.uint8))

            # Build mapping for every target frame to index in loaded_frames
            mapping = []
            for target_idx in range(expected_frames):
                target_ts = target_idx * time_per_frame
                # binary search for closest readable timestamp
                import bisect
                pos = bisect.bisect_left(readable_timestamps, target_ts)
                # choose closest between pos-1 and pos
                candidates = []
                if pos > 0:
                    candidates.append(pos - 1)
                if pos < len(readable_timestamps):
                    candidates.append(pos)
                best = candidates[0] if candidates else 0
                best_diff = abs(readable_timestamps[best] - target_ts) if candidates else float('inf')
                for c in candidates:
                    diff = abs(readable_timestamps[c] - target_ts)
                    if diff < best_diff:
                        best_diff = diff
                        best = c
                mapping.append(best)

            # Write frames according to mapping
            for m in mapping:
                frame_to_write = loaded_frames[m]
                out.write(frame_to_write)

        else:
            # Fallback behavior: replicate last valid frame strategy
            frame_idx = 0
            last_valid_frame = None
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for expected_idx in range(expected_frames):
                if expected_idx in valid_frames:
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        out.write(frame)
                        last_valid_frame = frame
                        frame_idx += 1
                else:
                    if last_valid_frame is not None:
                        out.write(last_valid_frame)
                    else:
                        black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                        out.write(black_frame)
        
        cap.release()
        out.release()
        
        # Keep the fixed file as a separate file (do not overwrite the original).
        # Return the path to the fixed file so callers can update selection if desired.
        try:
            print(f"Fixed video saved to: {output_path}")
        except Exception:
            pass
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
                
                video_path = self.fix_video_frames(video_path, metadata, valid_frames, timestamps=timestamps)
                
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
