"""Read/write persistent config (model path, file dialog history, etc.)."""

import json
import os
import sys
from pathlib import Path


def _get_repo_root() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent.parent


def _config_dir() -> Path:
    p = _get_repo_root() / "config"
    p.mkdir(exist_ok=True)
    return p


def _config_file_path() -> Path:
    return _config_dir() / "app_config.json"


def _read_config() -> dict:
    cf = _config_file_path()
    if not cf.exists():
        return _migrate_old_config()
    try:
        return json.loads(cf.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _migrate_old_config() -> dict:
    """Migrate from legacy config/track_model_path.json if it exists."""
    old = _config_dir() / "track_model_path.json"
    if not old.exists():
        return {}
    try:
        old_data = json.loads(old.read_text(encoding="utf-8"))
        path = old_data.get("track_model")
        if path and os.path.isfile(path):
            new_data = {"track_model": path}
            _write_config(new_data)
            return new_data
    except Exception:
        pass
    return {}


def _write_config(data: dict) -> None:
    _config_file_path().write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_path(key: str) -> str | None:
    """Return a saved path by key, or None if missing/invalid."""
    data = _read_config()
    path = data.get(key)
    if path and os.path.exists(path):
        return path
    return None


def save_path(key: str, path: str) -> None:
    """Persist a path under the given key. If path is a file, save its parent dir."""
    data = _read_config()
    if os.path.isfile(path):
        data[key] = os.path.dirname(path)
    else:
        data[key] = path
    _write_config(data)


def load_model_path() -> str | None:
    """Return the saved track model path, or None."""
    return load_path("track_model")


def save_model_path(model_path: str) -> None:
    """Persist the given track model file path."""
    data = _read_config()
    data["track_model"] = model_path
    _write_config(data)


def get_all_config() -> dict:
    """Return all config for the frontend."""
    data = _read_config()
    return {
        "track_model_path": data.get("track_model"),
        "last_recording_dir": data.get("last_recording_dir"),
        "last_model_dir": data.get("last_model_dir"),
    }
