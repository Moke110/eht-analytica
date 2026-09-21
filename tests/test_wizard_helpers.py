"""Light tests for the wizard helpers (generated data only, no GUI)."""

from __future__ import annotations

from pathlib import Path

import wizard_helpers as helpers


def test_default_download_base_format():
    url = helpers.default_download_base("v0.4.0")
    assert url == "https://github.com/Moke110/eht-analytica/releases/download/v0.4.0"


def test_default_install_dir_is_per_user(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert helpers.default_install_dir() == tmp_path / "Programs" / "EHT_Analytica"


def test_registry_uninstall_data_shape(tmp_path):
    data = helpers.registry_uninstall_data(tmp_path, "v0.4.0", 5_500_000_000)
    assert data["DisplayName"] == "EHT Analytica"
    assert data["DisplayVersion"] == "v0.4.0"
    assert data["InstallLocation"] == str(tmp_path)
    assert data["DisplayIcon"].endswith("EHT_Analytica.exe")
    # EstimatedSize is in KiB
    assert data["EstimatedSize"] == 5_500_000_000 // 1024
    assert data["NoModify"] == 1 and data["NoRepair"] == 1
    # Add/Remove Programs must relaunch the uninstaller, not the Setup wizard
    assert "--uninstall" in data["UninstallString"]
    assert str(tmp_path) in data["UninstallString"]
    assert data["UninstallString"].startswith('"')


def test_is_cancellable_event():
    assert helpers.is_cancellable_event("download")
    assert helpers.is_cancellable_event("extract")
    # The terminal event must never raise: the install is already complete
    assert not helpers.is_cancellable_event("done")


def test_shortcut_ps_command_quotes_paths():
    cmd = helpers.shortcut_ps_command(
        Path(r"C:\Users\a b\Desktop\EHT Analytica.lnk"),
        Path(r"C:\Programs\EHT_Analytica\EHT_Analytica.exe"),
        Path(r"C:\Programs\EHT_Analytica"),
    )
    assert "WScript.Shell" in cmd
    assert r"EHT Analytica.lnk" in cmd
    assert "TargetPath='C:\\Programs\\EHT_Analytica\\EHT_Analytica.exe'" in cmd
    assert "$s.Save()" in cmd


def test_ps_quote_escapes_single_quotes():
    assert helpers._ps_quote("it's") == "it''s"


def test_start_menu_dir_under_appdata(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    d = helpers.start_menu_dir()
    assert d == tmp_path / "Microsoft" / "Windows" / "Start Menu" / \
        "Programs" / "EHT Analytica"


def test_format_bytes():
    assert helpers.format_bytes(0) == "0 B"
    assert helpers.format_bytes(2048) == "2.0 KB"
    assert helpers.format_bytes(5 * 1024**3) == "5.0 GB"


def test_detect_nvidia_gpu_returns_shape():
    found, name = helpers.detect_nvidia_gpu()
    assert isinstance(found, bool)
    assert isinstance(name, str)
    if found:
        assert name  # a name is always reported when a GPU is found
