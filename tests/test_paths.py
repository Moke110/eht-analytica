"""Tests for dev/frozen path resolution (backend.utils.paths)."""

from __future__ import annotations

import sys
from pathlib import Path

import backend.utils.paths as paths


def test_dev_config_dir_is_app_root_config(monkeypatch):
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    assert paths.get_config_dir() == paths.get_app_root() / "config"


def test_frozen_windows_config_dir_uses_appdata(monkeypatch, tmp_path):
    fake_appdata = tmp_path / "AppData" / "Roaming"
    fake_appdata.mkdir(parents=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(fake_appdata))

    assert paths.get_config_dir() == fake_appdata / "EHT_Analytica"


def test_frozen_windows_config_dir_falls_back_without_appdata(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("APPDATA", raising=False)

    expected = Path.home() / "AppData" / "Roaming" / "EHT_Analytica"
    assert paths.get_config_dir() == expected


def test_frozen_windows_config_dir_is_outside_app_root(monkeypatch, tmp_path):
    """The Installation directory can be replaced freely; config survives."""
    fake_appdata = tmp_path / "AppData" / "Roaming"
    fake_appdata.mkdir(parents=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(fake_appdata))
    monkeypatch.setattr(sys, "executable", str(tmp_path / "Installation" / "EHT_Analytica.exe"))

    app_root = paths.get_app_root()
    config_dir = paths.get_config_dir()
    assert config_dir.is_relative_to(fake_appdata)
    assert not config_dir.is_relative_to(app_root)


def test_frozen_macos_config_dir_uses_application_support(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")

    expected = Path.home() / "Library" / "Application Support" / "EHT_Analytica"
    assert paths.get_config_dir() == expected
