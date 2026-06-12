"""Video session management — single VideoCapture open for metadata + first frame."""

from __future__ import annotations

import base64
import uuid

import cv2


class VideoSession:
    """Holds state for one opened video."""

    def __init__(self, video_path: str):
        self.video_id = uuid.uuid4().hex[:12]
        self.video_path = video_path
        self.metadata: dict = {}


# In-memory session store
_sessions: dict[str, VideoSession] = {}


def open_video(video_path: str) -> VideoSession:
    """Open video once, read metadata and first frame base64. Returns session."""
    sess = VideoSession(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    sess.metadata = {
        'path': video_path,
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        if cap.get(cv2.CAP_PROP_FPS) > 0 else 0,
    }

    ret, first_frame = cap.read()
    if ret and first_frame is not None:
        _, buf = cv2.imencode(".png", first_frame)
        sess.first_frame_base64 = base64.b64encode(buf).decode("utf-8")
    else:
        sess.first_frame_base64 = ""

    cap.release()

    _sessions[sess.video_id] = sess
    return sess


def get_session(video_id: str) -> VideoSession | None:
    return _sessions.get(video_id)
