"""Track API endpoints — video open/process, model load, tracking."""

import os

from fastapi import APIRouter, HTTPException, Request
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
from backend.services.video_service import open_video, get_session
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


@router.post("/model/load")
def model_load(req: ModelLoadRequest):
    # Validate model name against the central registry
    import json
    from backend.utils.paths import get_models_json_path
    models_json = get_models_json_path()
    if not models_json.exists():
        raise HTTPException(500, "models.json registry not found")
    with open(models_json, "r") as f:
        registry = json.load(f)
    valid_names = [m["name"] for m in registry.get("models", [])]
    if req.model_name not in valid_names:
        raise HTTPException(400, f"Unknown model: {req.model_name}. Available: {', '.join(valid_names)}")
    tid = load_model_async(req.model_name)
    return {"task_id": tid}


@router.get("/model/status", response_model=ModelStatusResponse)
def model_status():
    s = get_model_status()
    return ModelStatusResponse(**s)


@router.post("/start")
def track_start(req: TrackStartRequest):
    try:
        rois_dicts = [r.model_dump() for r in req.rois]
        tid = start_tracking(req.video_id, rois_dicts, req.output_folder, req.save_tracked_video, req.save_inferences)
        return {"task_id": tid}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/task/{task_id}")
def cancel_task(task_id: str):
    ok = task_manager.cancel(task_id)
    return {"cancelled": ok}


@router.get("/task/{task_id}/stream")
async def stream_task(task_id: str, request: Request):
    async def event_generator():
        async for data in task_manager.stream_events(task_id):
            if await request.is_disconnected():
                task_manager.cancel(task_id)
                return
            yield data
    return StreamingResponse(
        event_generator(),
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
