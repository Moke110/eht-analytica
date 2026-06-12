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
    model_name: str  # e.g. "unet_v2", "unet_v3"


class ModelInfo(BaseModel):
    name: str
    display_name: str
    description: str
    input_size: list[int]
    target_size: int
    num_peaks: int


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
    save_tracked_video: bool = False
    save_inferences: bool = False


# ── Analyze ────────────────────────────────────────────────────────────

class AnalyzeValidateRequest(BaseModel):
    paths: list[str]


class AnalyzeValidateResponse(BaseModel):
    valid: list[str]
    invalid: list[str]


class AnalyzeStartRequest(BaseModel):
    csv_paths: list[str]
    job_dir: Optional[str] = None


class AnalyzeSaveRequest(BaseModel):
    fs_results: list[dict]
    metrics_rows: Optional[list[dict]] = None
    output_folder: str
    job_dir: Optional[str] = None


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
    select_file: Optional[str] = None


class ConfigResponse(BaseModel):
    track_model_name: Optional[str] = None
    last_recording_dir: Optional[str] = None
    last_model_dir: Optional[str] = None
    roi_names: list[str] = Field(default_factory=list)


class InitJobDirRequest(BaseModel):
    video_path: str


class InitJobDirResponse(BaseModel):
    job_dir: str


class JobConfigResponse(BaseModel):
    job_dir: str
    samples: list[dict] = Field(default_factory=list)
    metadata_keys: list[str] = Field(default_factory=list)


# ── Reports ─────────────────────────────────────────────────────────────

class GroupMetricStats(BaseModel):
    sample_values: list[float] = Field(default_factory=list)
    mean: float = 0.0
    ste: float = 0.0


class ReportsPlotRequest(BaseModel):
    job_dir: str
    sample_ids: list[str] = Field(default_factory=list)
    group_keys: list[str] = Field(default_factory=list)


class ReportsPlotResponse(BaseModel):
    metrics: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)
    data: dict[str, dict[str, GroupMetricStats]] = Field(default_factory=dict)


# ── SSE / Task ─────────────────────────────────────────────────────────

class SSEEvent(BaseModel):
    event: Literal["progress", "complete", "error", "cancelled"]
    percent: float = 0
    message: str = ""
    result: Optional[dict] = None
    status: Literal["running", "completed", "failed", "cancelled"] = "running"
