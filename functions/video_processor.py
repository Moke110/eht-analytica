"""Video metadata utility for EHT Analytica."""

import cv2


def get_video_metadata(video_path):
    """Return dict of video metadata without storing any state."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    metadata = {
        'path': video_path,
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': (cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
                     if cap.get(cv2.CAP_PROP_FPS) > 0 else 0),
    }

    cap.release()
    return metadata
