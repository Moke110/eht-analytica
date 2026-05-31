"""Track API endpoints — video open/process, model load, tracking."""

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.models.schemas import (
    FileDialogRequest,
    FileDialogResponse,
    ModelLoadRequest,
    ModelStatusResponse,
    TrackStartRequest,
    VideoOpenRequest,
    VideoOpenResponse,
    VideoMetadata,
)
from backend.services.task_manager import task_manager
from backend.services.video_service import open_video, process_video_async, get_session
from backend.services.tracking_service import (
    get_model_status,
    load_model_async,
    start_tracking,
)

router = APIRouter(prefix="/api/track", tags=["track"])


@router.post("/video/open", response_model=VideoOpenResponse)
def video_open(req: VideoOpenRequest):
    if not os.path.isfile(req.path):
        raise HTTPException(400, f"Video not found: {req.path}")

    sess = open_video(req.path)
    meta = sess.metadata
    return VideoOpenResponse(
        video_id=sess.video_id,
        metadata=VideoMetadata(
            path=meta["path"],
            fps=meta["fps"],
            frame_count=meta["frame_count"],
            width=meta["width"],
            height=meta["height"],
            duration=meta["duration"],
        ),
        first_frame_base64=sess.first_frame_base64,
    )


@router.post("/video/process")
def video_process(req: VideoOpenRequest):
    """Open video and start frame scanning. Returns task_id for SSE."""
    if not os.path.isfile(req.path):
        raise HTTPException(400, f"Video not found: {req.path}")

    sess = open_video(req.path)
    tid = process_video_async(sess.video_id)
    return {"task_id": tid, "video_id": sess.video_id}


@router.post("/model/load")
def model_load(req: ModelLoadRequest):
    if not os.path.isfile(req.path):
        raise HTTPException(400, f"Model not found: {req.path}")
    tid = load_model_async(req.path)
    return {"task_id": tid}


@router.get("/model/status", response_model=ModelStatusResponse)
def model_status():
    s = get_model_status()
    return ModelStatusResponse(**s)


@router.post("/start")
def track_start(req: TrackStartRequest):
    try:
        rois_dicts = [r.model_dump() for r in req.rois]
        tid = start_tracking(req.video_id, rois_dicts, req.output_folder)
        return {"task_id": tid}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/task/{task_id}")
def cancel_task(task_id: str):
    ok = task_manager.cancel(task_id)
    return {"cancelled": ok}


@router.get("/task/{task_id}/stream")
async def stream_task(task_id: str):
    return StreamingResponse(
        task_manager.stream_events(task_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/native-file-dialog", response_model=FileDialogResponse)
def track_file_dialog(req: FileDialogRequest):
    """Convenience alias so the frontend can use /api/track/native-file-dialog."""
    from backend.api.system import _native_file_dialog
    paths = _native_file_dialog(req)
    return FileDialogResponse(paths=paths)
