"""Video session management — wraps VideoProcessor for the API layer."""

from __future__ import annotations

import base64
import uuid

import cv2

from functions.video_processor import VideoProcessor
from backend.services.task_manager import task_manager


class VideoSession:
    """Holds state for one opened video."""

    def __init__(self, video_path: str):
        self.video_id = uuid.uuid4().hex[:12]
        self.video_path = video_path
        self.metadata: dict = {}
        self.first_frame_base64: str = ""
        self.frame_buffer: list = []
        self.valid_frame_count: int = 0


# In-memory session store
_sessions: dict[str, VideoSession] = {}


def open_video(video_path: str) -> VideoSession:
    """Open a video and extract first frame + metadata. Returns immediately."""
    sess = VideoSession(video_path)

    proc = VideoProcessor()
    sess.metadata = proc.get_video_metadata(video_path)

    # Read first frame for immediate display
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()

    if ret and first_frame is not None:
        _, buf = cv2.imencode(".png", first_frame)
        sess.first_frame_base64 = base64.b64encode(buf).decode("utf-8")

    _sessions[sess.video_id] = sess
    return sess


def get_session(video_id: str) -> VideoSession | None:
    return _sessions.get(video_id)


def process_video_async(video_id: str) -> str:
    """Start frame scanning in background. Returns task_id for SSE streaming."""
    sess = _sessions.get(video_id)
    if sess is None:
        raise ValueError(f"Unknown video session: {video_id}")

    tid = task_manager.create("video_process")

    def worker():
        try:
            proc = VideoProcessor()

            def progress_cb(percent, message):
                task_manager.update(tid, percent, message)

            def completion_cb(metadata, first_frame, error, frame_buffer, valid_count):
                if error:
                    task_manager.fail(tid, error)
                else:
                    sess.frame_buffer = frame_buffer or []
                    sess.valid_frame_count = valid_count or 0
                    task_manager.complete(tid, {
                        "valid_frame_count": sess.valid_frame_count,
                        "duration": metadata.get("duration", 0) if metadata else 0,
                    })

            proc.process_video(sess.video_path, progress_cb, completion_cb)
        except Exception as e:
            task_manager.fail(tid, str(e))

    import threading
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return tid
