"""Read/write persistent config (model path, file dialog history, etc.)."""

import json
import os
import sys
import tempfile
from pathlib import Path


def _config_dir() -> Path:
    from backend.utils.paths import get_config_dir
    p = get_config_dir()
    p.mkdir(parents=True, exist_ok=True)
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
    """Write config atomically using tempfile + os.replace."""
    cf = _config_file_path()
    tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8",
        dir=cf.parent, delete=False,
        suffix=".tmp", prefix="app_config_",
    )
    try:
        tmp.write(json.dumps(data, indent=2, ensure_ascii=False))
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, str(cf))
    except Exception:
        tmp.close()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise


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


def load_model_name() -> str | None:
    """Return the saved track model name, or None.

    Migrates from legacy ``track_model`` (absolute path) to ``track_model_name``
    (short name like "unet_v3") if needed.
    """
    data = _read_config()
    if "track_model_name" in data:
        return data["track_model_name"]
    # Migration: derive model name from old absolute path
    old_path = data.get("track_model", "")
    if old_path:
        name = _derive_model_name_from_path(old_path)
        if name:
            data["track_model_name"] = name
            _write_config(data)
            return name
    return None


def save_model_name(model_name: str) -> None:
    """Persist the given track model name (e.g. "unet_v3")."""
    data = _read_config()
    data["track_model_name"] = model_name
    _write_config(data)


def _derive_model_name_from_path(path: str) -> str | None:
    """Guess model name from old track_model absolute path."""
    norm = os.path.normpath(path).replace("\\", "/").lower()
    if "/unet_v3/" in norm or norm.endswith("/unet_v3"):
        return "unet_v3"
    return None


def get_all_config() -> dict:
    """Return all config for the frontend."""
    data = _read_config()
    return {
        "track_model_name": data.get("track_model_name") or "unet_v3",
        "last_recording_dir": data.get("last_recording_dir"),
        "last_model_dir": data.get("last_model_dir"),
        "roi_names": data.get("roi_names", []),
    }


def load_roi_names() -> list[str]:
    return _read_config().get("roi_names", [])


def save_roi_names(names: list[str]) -> None:
    data = _read_config()
    data["roi_names"] = names
    _write_config(data)
