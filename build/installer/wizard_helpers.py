"""Pure helpers for the Setup wizard — no tkinter imports here.

Everything Windows-shell related (shortcuts via PowerShell, the uninstall
registry entry, GPU detection, self-deleting uninstall scheduling) lives in
this module so the GUI stays a thin wrapper and the helpers stay testable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

APP_NAME = "EHT_Analytica"
DISPLAY_NAME = "EHT Analytica"
PUBLISHER = "Chang Wang"
GITHUB_REPO = "Moke110/eht-analytica"
UNINSTALL_REGISTRY_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\EHT_Analytica"
CREATE_NO_WINDOW = 0x08000000


def default_install_dir() -> Path:
    """Per-user install location (no admin rights required)."""
    local = os.environ.get("LOCALAPPDATA")
    base = Path(local) if local else Path.home() / "AppData" / "Local"
    return base / "Programs" / APP_NAME


def default_download_base(tag: str) -> str:
    """Official GitHub release download base for the given tag."""
    return f"https://github.com/{GITHUB_REPO}/releases/download/{tag}"


def detect_nvidia_gpu() -> tuple[bool, str]:
    """Return (found, gpu_name) via nvidia-smi; informational only."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return False, ""
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
            creationflags=CREATE_NO_WINDOW,
        )
        if out.returncode == 0 and out.stdout.strip():
            return True, out.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return False, ""


def _ps_quote(value: str) -> str:
    """Escape a value for a single-quoted PowerShell string."""
    return value.replace("'", "''")


def shortcut_ps_command(lnk_path: Path, target_path: Path,
                        working_dir: Path) -> str:
    """PowerShell command that creates a .lnk shortcut (WScript.Shell COM)."""
    return (
        f"$s=(New-Object -ComObject WScript.Shell)"
        f".CreateShortcut('{_ps_quote(str(lnk_path))}');"
        f"$s.TargetPath='{_ps_quote(str(target_path))}';"
        f"$s.WorkingDirectory='{_ps_quote(str(working_dir))}';"
        f"$s.IconLocation='{_ps_quote(str(target_path))},0';"
        f"$s.Save()"
    )


def _run_ps(command: str, timeout: float = 30.0) -> None:
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
        check=True, capture_output=True, timeout=timeout,
        creationflags=CREATE_NO_WINDOW,
    )


def desktop_dir() -> Path:
    """The user's real Desktop (OneDrive-redirected safe)."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "[Environment]::GetFolderPath('Desktop')"],
            capture_output=True, text=True, timeout=15, check=True,
            creationflags=CREATE_NO_WINDOW,
        )
        p = Path(out.stdout.strip())
        if p.exists():
            return p
    except Exception:
        pass
    return Path.home() / "Desktop"


def start_menu_dir() -> Path:
    """Per-user Start Menu program group for the app."""
    appdata = os.environ.get("APPDATA")
    base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    return base / "Microsoft" / "Windows" / "Start Menu" / "Programs" / DISPLAY_NAME


def create_shortcut(lnk_path: Path, target_path: Path, working_dir: Path) -> None:
    _run_ps(shortcut_ps_command(lnk_path, target_path, working_dir))


def registry_uninstall_data(install_dir: Path, version: str,
                            estimated_size_bytes: int) -> dict:
    """Values written to the HKCU Add/Remove Programs entry."""
    install_dir = Path(install_dir)
    uninstall_exe = install_dir / "Uninstall.exe"
    return {
        "DisplayName": DISPLAY_NAME,
        "DisplayVersion": version,
        "Publisher": PUBLISHER,
        "InstallLocation": str(install_dir),
        "DisplayIcon": str(install_dir / f"{APP_NAME}.exe"),
        # The uninstaller is this same exe; it needs the flag to run in
        # uninstall mode instead of opening the Setup wizard again.
        "UninstallString": f'"{uninstall_exe}" --uninstall --install-dir '
                           f'"{install_dir}"',
        "EstimatedSize": max(1, estimated_size_bytes // 1024),
        "NoModify": 1,
        "NoRepair": 1,
    }


def write_uninstall_registry(data: dict) -> None:
    import winreg
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, UNINSTALL_REGISTRY_KEY,
                            0, winreg.KEY_SET_VALUE) as key:
        for name, value in data.items():
            reg_type = winreg.REG_SZ if isinstance(value, str) else winreg.REG_DWORD
            winreg.SetValueEx(key, name, 0, reg_type, value)


def remove_uninstall_registry() -> None:
    import winreg
    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, UNINSTALL_REGISTRY_KEY)
    except FileNotFoundError:
        pass


def is_cancellable_event(kind: str) -> bool:
    """Whether a cancel request must abort on this progress event kind.

    The terminal ``done`` event must never raise: the payload is already
    fully assembled and the post-install steps still need to run.
    """
    return kind != "done"


def schedule_self_uninstall(install_dir: Path) -> None:
    """Remove the Installation (and this running uninstaller exe) on exit.

    A running exe cannot be deleted, but it can be moved: wait for this
    process to exit, move the exe to %TEMP%, remove the directory, then
    delete the moved exe. Paths travel as environment variables so they
    never need to be interpolated into the cmd string.
    """
    install_dir = Path(install_dir)
    temp_exe = Path(os.environ.get("TEMP", str(Path.home()))) / \
        f"{APP_NAME}_uninstaller.exe"
    env = dict(
        os.environ,
        EHT_UNINST_SELF=str(sys.executable),
        EHT_UNINST_TMP=str(temp_exe),
        EHT_UNINST_DIR=str(install_dir),
    )
    command = (
        'timeout /t 2 /nobreak >nul'
        ' & move /y "%EHT_UNINST_SELF%" "%EHT_UNINST_TMP%"'
        ' & rmdir /s /q "%EHT_UNINST_DIR%"'
        ' & del /q "%EHT_UNINST_TMP%"'
    )
    subprocess.Popen(["cmd", "/c", command], shell=False, env=env,
                     creationflags=CREATE_NO_WINDOW)


def format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024  # type: ignore[assignment]
    return f"{n} B"


def launch_app(install_dir: Path) -> None:
    """Start the installed application, detached from the installer."""
    exe = Path(install_dir) / f"{APP_NAME}.exe"
    subprocess.Popen([str(exe)], cwd=str(Path(install_dir)),
                     creationflags=CREATE_NO_WINDOW)
