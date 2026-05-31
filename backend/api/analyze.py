"""Analyze API endpoints — CSV validation, analysis, save results."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.models.schemas import (
    AnalyzeValidateRequest,
    AnalyzeValidateResponse,
    AnalyzeStartRequest,
    AnalyzeSaveRequest,
    AnalyzeSaveResponse,
)
from backend.services.task_manager import task_manager
from backend.services.analysis_service import validate_paths, start_analysis, save_results

router = APIRouter(prefix="/api/analyze", tags=["analyze"])


@router.post("/validate", response_model=AnalyzeValidateResponse)
def validate(req: AnalyzeValidateRequest):
    valid, invalid = validate_paths(req.paths)
    return AnalyzeValidateResponse(valid=valid, invalid=invalid)


@router.post("/start")
def analyze_start(req: AnalyzeStartRequest):
    if not req.csv_paths:
        raise HTTPException(400, "No CSV paths provided")
    tid = start_analysis(req.csv_paths)
    return {"task_id": tid}


@router.post("/save", response_model=AnalyzeSaveResponse)
def analyze_save(req: AnalyzeSaveRequest):
    result = save_results(req.fs_results, req.metrics_rows, req.output_folder)
    return AnalyzeSaveResponse(**result)


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
