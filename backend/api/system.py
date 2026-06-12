"""System endpoints: health, config, native file dialogs, open-folder."""

import glob
import os
import platform
import subprocess
import sys
import tkinter.filedialog
import tkinter

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.models.schemas import (
    ConfigResponse,
    FileDialogRequest,
    FileDialogResponse,
    InitJobDirRequest,
    InitJobDirResponse,
    JobConfigResponse,
    OpenFolderRequest,
    ReportsPlotRequest,
    ReportsPlotResponse,
)
from backend.utils.config import save_path, get_all_config, save_roi_names, load_path
from backend.services.job_config import (
    init_job_dir,
    get_job_config,
    resolve_path,
    save_metadata,
)

router = APIRouter(prefix="/api/system", tags=["system"])


class ListCsvRequest(BaseModel):
    folder: str


class ListCsvResponse(BaseModel):
    paths: list[str]


def _native_file_dialog(req: FileDialogRequest) -> list[str]:
    """Open a native OS file/folder dialog using tkinter (bundled with Python)."""
    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    initialdir = req.initial_path if req.initial_path and os.path.isdir(req.initial_path) else None

    try:
        if req.type == "directory":
            path = tkinter.filedialog.askdirectory(title=req.title, initialdir=initialdir)
            return [path] if path else []

        filetypes = [(label, " ".join(patterns)) for label, patterns in req.filters] if req.filters else []
        if req.multi:
            paths = tkinter.filedialog.askopenfilenames(title=req.title, filetypes=filetypes, initialdir=initialdir)
            return list(paths)
        else:
            path = tkinter.filedialog.askopenfilename(title=req.title, filetypes=filetypes, initialdir=initialdir)
            return [path] if path else []
    finally:
        root.destroy()


@router.post("/native-file-dialog", response_model=FileDialogResponse)
def open_native_file_dialog(req: FileDialogRequest):
    paths = _native_file_dialog(req)
    return FileDialogResponse(paths=paths)


@router.post("/open-folder")
def open_folder(req: OpenFolderRequest):
    p = req.path
    sel_file = None
    if req.select_file:
        sel_file = os.path.join(req.path, req.select_file)
        if os.path.isfile(sel_file):
            p = sel_file
    if not os.path.isdir(p):
        p = os.path.dirname(p)
    if not os.path.isdir(p):
        return {"opened": False, "error": f"Not a directory: {p}"}

    system = platform.system()
    try:
        if system == "Windows":
            if sel_file and os.path.isfile(sel_file):
                subprocess.Popen(["explorer", "/select,", os.path.normpath(sel_file)])
            else:
                os.startfile(p)
        elif system == "Darwin":
            subprocess.Popen(["open", "-R", sel_file] if sel_file and os.path.isfile(sel_file) else ["open", p])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(sel_file) if sel_file and os.path.isfile(sel_file) else p])
        return {"opened": True}
    except Exception as e:
        return {"opened": False, "error": str(e)}


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/list-csv", response_model=ListCsvResponse)
def list_csv(req: ListCsvRequest):
    folder = req.folder
    if not os.path.isdir(folder):
        return ListCsvResponse(paths=[])
    csvs = glob.glob(os.path.join(folder, "*.csv"))
    return ListCsvResponse(paths=sorted(csvs))


class SavePathRequest(BaseModel):
    key: str
    path: str


class LoadPathRequest(BaseModel):
    key: str


class SaveRoiNamesRequest(BaseModel):
    names: list[str]


@router.get("/config", response_model=ConfigResponse)
def get_config():
    return ConfigResponse(**get_all_config())


@router.get("/models")
def list_models():
    """Return available models from the central models.json registry."""
    import json
    from pathlib import Path
    models_json = Path(__file__).resolve().parent.parent.parent / "model" / "models.json"
    if not models_json.exists():
        return {"models": []}
    with open(models_json, "r") as f:
        registry = json.load(f)
    return {
        "models": [
            {
                "name": m["name"],
                "display_name": m.get("display_name", m["name"]),
                "description": m.get("description", ""),
                "input_size": m.get("input_size", [512, 512]),
                "target_size": m.get("target_size", 512),
                "num_peaks": m.get("num_peaks", 2),
            }
            for m in registry.get("models", [])
        ]
    }


@router.get("/device")
def device_info():
    """Return system device capability info."""
    import torch
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_mb = props.total_memory / (1024 ** 2)
            return {
                "device": "cuda",
                "device_info": f"{props.name} ({total_mb:.0f} MB)",
            }
    except Exception:
        pass
    import psutil
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024 ** 3)
    return {
        "device": "cpu",
        "device_info": f"CPU ({total_gb:.1f} GB RAM)",
    }


@router.post("/config/save-path")
def save_path_endpoint(req: SavePathRequest):
    save_path(req.key, req.path)
    return {"ok": True}


@router.post("/config/load-path")
def load_path_endpoint(req: LoadPathRequest):
    return {"path": load_path(req.key)}


@router.post("/config/save-roi-names")
def save_roi_names_endpoint(req: SaveRoiNamesRequest):
    save_roi_names(req.names)
    return {"ok": True}


@router.post("/init-job-dir", response_model=InitJobDirResponse)
def init_job_dir_endpoint(req: InitJobDirRequest):
    jd = init_job_dir(req.video_path)
    return InitJobDirResponse(job_dir=jd)


class JobConfigQuery(BaseModel):
    job_dir: str


class SaveMetadataRequest(BaseModel):
    job_dir: str
    metadata_keys: list[str] = []
    samples: list[dict] = []


@router.post("/job-config", response_model=JobConfigResponse)
def get_job_config_endpoint(req: JobConfigQuery):
    config = get_job_config(req.job_dir)
    return JobConfigResponse(
        job_dir=req.job_dir,
        samples=config.get("samples", []),
        metadata_keys=config.get("metadata_keys", []),
    )


@router.post("/save-metadata")
def save_metadata_endpoint(req: SaveMetadataRequest):
    save_metadata(req.job_dir, req.metadata_keys, req.samples)
    return {"ok": True}


class SaveFileRequest(BaseModel):
    dir: str
    filename: str
    content: str  # plain text, or base64 data URL for binary


@router.post("/save-file")
def save_file(req: SaveFileRequest):
    import base64

    safe = os.path.basename(req.filename.replace("\\", "/"))
    if not safe:
        raise HTTPException(400, "Invalid filename")

    os.makedirs(req.dir, exist_ok=True)
    fpath = os.path.join(req.dir, safe)

    if req.content.startswith("data:"):
        # data URL — decode and write binary
        header, b64 = req.content.split(",", 1)
        try:
            data = base64.b64decode(b64)
        except Exception:
            raise HTTPException(400, "Invalid base64 content")
        with open(fpath, "wb") as f:
            f.write(data)
    else:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(req.content)

    return {"ok": True, "path": fpath}


@router.post("/reports/plot-data", response_model=ReportsPlotResponse)
def reports_plot_data(req: ReportsPlotRequest):
    from backend.services.reports_service import compute_plot_data
    return compute_plot_data(req.job_dir, req.sample_ids, req.group_keys)


@router.post("/reports/full-data")
def reports_full_data(req: ReportsPlotRequest):
    from backend.services.reports_service import compute_full_data
    return compute_full_data(req.job_dir, req.sample_ids)
