"""Pydantic request/response models for the EHT Analytica API."""

from pydantic import BaseModel, Field
from typing import Optional, Literal


# ── Video ──────────────────────────────────────────────────────────────

class VideoOpenRequest(BaseModel):
    path: str


class VideoMetadata(BaseModel):
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration: float


class VideoOpenResponse(BaseModel):
    video_id: str
    metadata: VideoMetadata
    first_frame_base64: str


# ── ROI ────────────────────────────────────────────────────────────────

class RoiDefinition(BaseModel):
    name: str
    x: int
    y: int
    width: int
    height: int
    color: tuple[int, int, int]


# ── Model ──────────────────────────────────────────────────────────────

class ModelLoadRequest(BaseModel):
    path: str


class ModelStatusResponse(BaseModel):
    loaded: bool
    model_name: Optional[str] = None
    model_path: Optional[str] = None
    device: Optional[str] = None
    device_info: Optional[str] = None


# ── Track ──────────────────────────────────────────────────────────────

class TrackStartRequest(BaseModel):
    video_id: str
    rois: list[RoiDefinition]
    output_folder: str


# ── Analyze ────────────────────────────────────────────────────────────

class AnalyzeValidateRequest(BaseModel):
    paths: list[str]


class AnalyzeValidateResponse(BaseModel):
    valid: list[str]
    invalid: list[str]


class AnalyzeStartRequest(BaseModel):
    csv_paths: list[str]


class AnalyzeSaveRequest(BaseModel):
    fs_results: list[dict]
    metrics_rows: Optional[list[dict]] = None
    output_folder: str


class AnalyzeSaveResponse(BaseModel):
    saved_files: list[str]
    metrics_path: Optional[str] = None


# ── System ─────────────────────────────────────────────────────────────

class FileDialogRequest(BaseModel):
    type: Literal["file", "directory"]
    title: str = "Select file"
    filters: list[list[str]] = Field(default_factory=list)
    multi: bool = False
    initial_path: Optional[str] = None


class FileDialogResponse(BaseModel):
    paths: list[str]


class OpenFolderRequest(BaseModel):
    path: str


class ConfigResponse(BaseModel):
    track_model_path: Optional[str] = None
    last_recording_dir: Optional[str] = None
    last_model_dir: Optional[str] = None


# ── SSE / Task ─────────────────────────────────────────────────────────

class SSEEvent(BaseModel):
    event: Literal["progress", "complete", "error", "cancelled"]
    percent: float = 0
    message: str = ""
    result: Optional[dict] = None
    status: Literal["running", "completed", "failed", "cancelled"] = "running"
