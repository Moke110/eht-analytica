"""System endpoints: health, config, native file dialogs, open-folder."""

import glob
import os
import platform
import subprocess
import sys
import tkinter.filedialog
import tkinter

from fastapi import APIRouter
from pydantic import BaseModel

from backend.models.schemas import (
    ConfigResponse,
    FileDialogRequest,
    FileDialogResponse,
    OpenFolderRequest,
)
from backend.utils.config import load_model_path, save_path, get_all_config

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
    if not os.path.isdir(p):
        p = os.path.dirname(p)
    if not os.path.isdir(p):
        return {"opened": False, "error": f"Not a directory: {p}"}

    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(p)
        elif system == "Darwin":
            subprocess.Popen(["open", p])
        else:
            subprocess.Popen(["xdg-open", p])
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


@router.get("/config", response_model=ConfigResponse)
def get_config():
    return ConfigResponse(**get_all_config())


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
