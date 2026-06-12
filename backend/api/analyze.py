"""Analyze API endpoints — CSV validation, analysis, save results, read CSVs for visualization."""

import csv
import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

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


class ReadCsvRequest(BaseModel):
    job_dir: str
    paths: list[str]  # relative paths within job_dir


class CsvData(BaseModel):
    filename: str
    columns: list[str]
    rows: list[list]


class ReadCsvResponse(BaseModel):
    files: list[CsvData]


@router.post("/read-csv", response_model=ReadCsvResponse)
def read_csv(req: ReadCsvRequest):
    """Read CSV files from a job directory and return contents as JSON."""
    jd = os.path.abspath(req.job_dir)
    files = []
    for rel_path in req.paths:
        abs_path = os.path.normpath(os.path.join(jd, rel_path))
        if not abs_path.startswith(jd + os.sep):
            raise HTTPException(403, f"Path traversal rejected: {rel_path}")
        if not abs_path.lower().endswith('.csv'):
            raise HTTPException(400, f"Not a CSV file: {rel_path}")
        if not os.path.isfile(abs_path):
            raise HTTPException(404, f"File not found: {rel_path}")

        rows = []
        with open(abs_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            columns = next(reader, [])
            for row in reader:
                rows.append(row)
        files.append(CsvData(filename=os.path.basename(abs_path),
                             columns=columns, rows=rows))
    return ReadCsvResponse(files=files)


@router.post("/validate", response_model=AnalyzeValidateResponse)
def validate(req: AnalyzeValidateRequest):
    valid, invalid = validate_paths(req.paths)
    return AnalyzeValidateResponse(valid=valid, invalid=invalid)


@router.post("/start")
def analyze_start(req: AnalyzeStartRequest):
    if not req.csv_paths:
        raise HTTPException(400, "No CSV paths provided")
    missing = [p for p in req.csv_paths if not os.path.isfile(p)]
    if missing:
        raise HTTPException(400, f"File(s) not found: {', '.join(missing)}")
    try:
        tid = start_analysis(req.csv_paths, req.job_dir)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))
    return {"task_id": tid}


@router.post("/save", response_model=AnalyzeSaveResponse)
def analyze_save(req: AnalyzeSaveRequest):
    result = save_results(req.fs_results, req.metrics_rows, req.output_folder, req.job_dir)
    return AnalyzeSaveResponse(**result)


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


@router.delete("/task/{task_id}")
def cancel_task(task_id: str):
    task_manager.cancel(task_id)
    return {"ok": True}
