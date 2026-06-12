"""Centralized path resolution for dev and frozen (PyInstaller) environments.

When frozen, the app root is the directory containing the EXE (where model/
and config/ live alongside).  In development, it's the project root derived
from this file's location.
"""

from __future__ import annotations

import sys
from pathlib import Path


def get_app_root() -> Path:
    """Return the application root directory.

    - Frozen (PyInstaller): the directory containing the EXE.
    - Dev: the project root (3 levels up from *backend/utils/paths.py*).
    """
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent.parent


def get_model_dir() -> Path:
    """``<app_root>/model`` — external model files."""
    return get_app_root() / "model"


def get_config_dir() -> Path:
    """``<app_root>/config`` — persistent app config (writable)."""
    return get_app_root() / "config"


def get_models_json_path() -> Path:
    """``<model_dir>/models.json`` — central model registry."""
    return get_model_dir() / "models.json"
