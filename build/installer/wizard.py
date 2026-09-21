"""EHT Analytica Setup wizard — tkinter GUI over the headless installer core.

Wizard flow: welcome (GPU info) -> options (install dir, shortcuts, proxy)
-> progress (per-Volume download + extraction) -> finish (launch).

Also serves as the uninstaller when started with ``--uninstall``: the
Installer drops a copy of itself as ``Uninstall.exe`` inside the
Installation and points the Add/Remove-Programs entry at it.

The volume manifest is embedded in the exe at build time (PyInstaller
datas); a side-car ``manifest.json`` next to this script is the dev-mode
fallback. Default download base is the official GitHub release URL; the
user may enter an explicit proxy (e.g. Clash Verge at ``127.0.0.1:7897``)
to route downloads, otherwise the Windows system proxy is used.
"""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import sys
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import wizard_helpers as helpers
from installer_core import (
    InstallCancelled,
    InstallError,
    Manifest,
    install_from_manifest,
)

WINDOW_TITLE = f"{helpers.DISPLAY_NAME} Setup"
PAD = 16


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_embedded_manifest() -> Manifest:
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        candidates.append(Path(getattr(sys, "_MEIPASS", "")) / "manifest.json")
    candidates.append(Path(__file__).resolve().parent / "manifest.json")
    for c in candidates:
        if c.is_file():
            return Manifest.from_dict(
                json.loads(c.read_text(encoding="utf-8")))
    raise SystemExit(
        "manifest.json not found (embedded or next to the wizard). "
        "Build via build/make_release.py so the manifest is embedded."
    )


# ---------------------------------------------------------------------------
# Uninstall mode
# ---------------------------------------------------------------------------

def run_uninstall(install_dir: Path) -> int:
    root = tk.Tk()
    root.withdraw()
    if not messagebox.askyesno(
        f"Uninstall {helpers.DISPLAY_NAME}",
        f"Remove {helpers.DISPLAY_NAME} from\n{install_dir}?\n\n"
        f"Your settings (under %APPDATA%\\{helpers.APP_NAME}) are kept.",
    ):
        return 0
    # Remove shortcuts
    desktop_lnk = helpers.desktop_dir() / f"{helpers.DISPLAY_NAME}.lnk"
    menu_lnk = helpers.start_menu_dir() / f"{helpers.DISPLAY_NAME}.lnk"
    desktop_lnk.unlink(missing_ok=True)
    menu_lnk.unlink(missing_ok=True)
    try:
        helpers.start_menu_dir().rmdir()  # only if now empty
    except OSError:
        pass
    helpers.remove_uninstall_registry()
    messagebox.showinfo(f"Uninstall {helpers.DISPLAY_NAME}",
                        "Uninstalled. The program folder will be removed "
                        "momentarily.")
    helpers.schedule_self_uninstall(install_dir)
    return 0


# ---------------------------------------------------------------------------
# Wizard
# ---------------------------------------------------------------------------

class WizardApp:
    def __init__(self, root: tk.Tk, manifest: Manifest, cli_base_url: str | None,
                 cli_install_dir: str | None,
                 cli_proxy: str | None = None) -> None:
        self.root = root
        self.manifest = manifest
        self.default_base = cli_base_url or helpers.default_download_base(
            manifest.tag)
        self.default_proxy = cli_proxy or ""
        self.q: queue.Queue[tuple[str, dict]] = queue.Queue()
        self.cancel_requested = False
        self.worker: threading.Thread | None = None
        self.install_dir = Path(cli_install_dir) if cli_install_dir else \
            helpers.default_install_dir()
        # Post-install options captured on the options screen
        self.opt_desktop = tk.BooleanVar(value=True)
        self.opt_start_menu = tk.BooleanVar(value=True)
        self.opt_launch = tk.BooleanVar(value=True)

        root.title(WINDOW_TITLE)
        root.minsize(560, 380)
        root.resizable(False, False)
        self.container = ttk.Frame(root, padding=PAD)
        self.container.pack(fill="both", expand=True)
        self._show_welcome()
        self.root.after(100, self._poll)

    # -- screen switching --------------------------------------------------

    def _clear(self) -> None:
        for child in self.container.winfo_children():
            child.destroy()

    def _show_welcome(self) -> None:
        self._clear()
        frame = ttk.Frame(self.container)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text=helpers.DISPLAY_NAME,
                  font=("Segoe UI", 20, "bold")).pack(pady=(24, 4))
        ttk.Label(frame, text=f"Version {self.manifest.tag}",
                  font=("Segoe UI", 10)).pack()
        ttk.Label(
            frame,
            text=f"This wizard will install {helpers.DISPLAY_NAME} "
                 f"({self._pretty_total()}) on your computer.",
            wraplength=480, justify="center",
        ).pack(pady=(20, 4))

        gpu_found, gpu_name = helpers.detect_nvidia_gpu()
        if gpu_found:
            info = f"NVIDIA GPU detected: {gpu_name} — GPU acceleration enabled."
            ttk.Label(frame, text=info, foreground="#0a7a0a").pack(pady=6)
        else:
            ttk.Label(
                frame,
                text="No NVIDIA GPU detected — the application will run in "
                     "CPU mode (slower, but fully functional).",
                foreground="#a06000", wraplength=480, justify="center",
            ).pack(pady=6)

        btns = ttk.Frame(frame)
        btns.pack(side="bottom", fill="x", pady=(16, 0))
        ttk.Button(btns, text="Cancel", command=self.root.destroy)\
            .pack(side="right", padx=(8, 0))
        ttk.Button(btns, text="Next >", command=self._show_options)\
            .pack(side="right")

    def _show_options(self) -> None:
        self._clear()
        frame = ttk.Frame(self.container)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Installation options",
                  font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 12))

        ttk.Label(frame, text="Install directory:").pack(anchor="w")
        dir_row = ttk.Frame(frame)
        dir_row.pack(fill="x", pady=(2, 8))
        self.dir_var = tk.StringVar(value=str(self.install_dir))
        ttk.Entry(dir_row, textvariable=self.dir_var).pack(
            side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(dir_row, text="Browse...",
                   command=self._browse_dir).pack(side="right")

        ttk.Checkbutton(frame, text="Create desktop shortcut",
                        variable=self.opt_desktop).pack(anchor="w", pady=2)
        ttk.Checkbutton(frame, text="Create Start Menu shortcut",
                        variable=self.opt_start_menu).pack(anchor="w", pady=2)
        ttk.Checkbutton(frame, text="Launch after installation completes",
                        variable=self.opt_launch).pack(anchor="w", pady=2)

        # Advanced: optional explicit proxy (Clash/V2Ray etc.)
        advanced = ttk.LabelFrame(frame, text="Advanced (optional)", padding=8)
        advanced.pack(fill="x", pady=(16, 0))
        ttk.Label(
            advanced,
            text="Proxy server used to download the payload. Leave empty to "
                 "use the Windows system proxy.\n"
                 "Example (Clash Verge): http://127.0.0.1:7897",
            wraplength=460, justify="left",
        ).pack(anchor="w")
        self.proxy_var = tk.StringVar(value=self.default_proxy)
        ttk.Entry(advanced, textvariable=self.proxy_var).pack(
            fill="x", pady=(2, 0))

        btns = ttk.Frame(frame)
        btns.pack(side="bottom", fill="x", pady=(16, 0))
        ttk.Button(btns, text="< Back",
                   command=self._show_welcome).pack(side="left")
        ttk.Button(btns, text="Cancel", command=self.root.destroy)\
            .pack(side="right", padx=(8, 0))
        ttk.Button(btns, text="Install", command=self._start_install)\
            .pack(side="right")

    def _show_progress(self) -> None:
        self._clear()
        frame = ttk.Frame(self.container)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Installing…",
                  font=("Segoe UI", 14, "bold")).pack(anchor="w")
        self.status_var = tk.StringVar(value="Starting…")
        ttk.Label(frame, textvariable=self.status_var,
                  wraplength=500).pack(anchor="w", pady=(12, 4))
        self.bar = ttk.Progressbar(frame, length=500, maximum=100)
        self.bar.pack(pady=4)
        self.detail_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.detail_var,
                  foreground="#606060").pack(anchor="w")
        btns = ttk.Frame(frame)
        btns.pack(side="bottom", fill="x", pady=(16, 0))
        ttk.Button(btns, text="Cancel",
                   command=self._cancel).pack(side="right")

    def _show_finish(self, ok: bool, message: str = "") -> None:
        self._clear()
        frame = ttk.Frame(self.container)
        frame.pack(fill="both", expand=True)
        if ok:
            ttk.Label(frame, text="Installation complete",
                      font=("Segoe UI", 16, "bold"),
                      foreground="#0a7a0a").pack(pady=(16, 8))
            ttk.Label(frame, text=f"{helpers.DISPLAY_NAME} "
                                  f"{self.manifest.tag} was installed to:\n"
                                  f"{self.install_dir}",
                      wraplength=480, justify="center").pack()
            ttk.Button(frame, text="Finish", command=self._finish)\
                .pack(side="bottom", pady=24)
        else:
            ttk.Label(frame, text="Installation failed",
                      font=("Segoe UI", 16, "bold"),
                      foreground="#b00020").pack(pady=(16, 8))
            ttk.Label(frame, text=message, wraplength=480,
                      justify="left").pack()
            row = ttk.Frame(frame)
            row.pack(side="bottom", pady=16)
            ttk.Button(row, text="Retry", command=self._show_options)\
                .pack(side="right", padx=(8, 0))
            ttk.Button(row, text="Close", command=self.root.destroy)\
                .pack(side="right")

    # -- actions -------------------------------------------------------------

    def _browse_dir(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.dir_var.get())
        if chosen:
            self.dir_var.set(chosen)

    def _start_install(self) -> None:
        if not self.manifest.volumes:
            messagebox.showerror(WINDOW_TITLE,
                                 "This build has no payload volumes "
                                 "(corrupt Setup exe).")
            return
        self.install_dir = Path(self.dir_var.get()).expanduser()
        base_url = self.default_base
        proxy = self.proxy_var.get().strip()
        # Capture option values on the main thread: tkinter variables must
        # not be touched from the worker thread.
        self.opts = {
            "desktop": bool(self.opt_desktop.get()),
            "start_menu": bool(self.opt_start_menu.get()),
            "launch": bool(self.opt_launch.get()),
        }
        # Cheap up-front disk space check (core re-checks authoritatively)
        largest = max(v.size_bytes for v in self.manifest.volumes)
        required = int((self.manifest.total_bytes + largest) * 1.05)
        try:
            free = self._free_bytes(self.install_dir)
        except OSError as e:
            messagebox.showerror(WINDOW_TITLE, f"Cannot check disk: {e}")
            return
        if free < required:
            messagebox.showerror(
                WINDOW_TITLE,
                f"Not enough disk space: need about "
                f"{helpers.format_bytes(required)}, only "
                f"{helpers.format_bytes(free)} free.",
            )
            return

        self._show_progress()
        self.cancel_requested = False
        self.worker = threading.Thread(
            target=self._worker,
            args=(base_url, self.install_dir, self.opts, proxy),
            daemon=True)
        self.worker.start()

    def _worker(self, base_url: str, install_dir: Path, opts: dict,
                proxy: str) -> None:
        try:
            install_from_manifest(self.manifest, base_url, install_dir,
                                  progress=self._emit, max_attempts=3,
                                  proxy=proxy)
            self._post_install(install_dir, opts)
            self.q.put(("done", {}))
        except InstallCancelled:
            self.q.put(("cancelled", {}))
        except InstallError as e:
            self.q.put(("error", {"message": str(e)}))
        except Exception as e:  # defensive: surface unexpected crashes
            self.q.put(("error", {"message": f"{e}\n{traceback.format_exc()}"}))

    def _post_install(self, install_dir: Path, opts: dict) -> None:
        """Side effects after assembly: uninstaller, shortcuts, registry."""
        # Uninstaller: copy of this exe (frozen builds only)
        if getattr(sys, "frozen", False):
            shutil.copy2(sys.executable, install_dir / "Uninstall.exe")
        # Shortcuts
        exe = install_dir / f"{helpers.APP_NAME}.exe"
        if opts["desktop"]:
            helpers.create_shortcut(
                helpers.desktop_dir() / f"{helpers.DISPLAY_NAME}.lnk",
                exe, install_dir)
        if opts["start_menu"]:
            helpers.create_shortcut(
                helpers.start_menu_dir() / f"{helpers.DISPLAY_NAME}.lnk",
                exe, install_dir)
        # Add/Remove Programs entry
        helpers.write_uninstall_registry(helpers.registry_uninstall_data(
            install_dir, self.manifest.tag, self.manifest.total_bytes))

    def _emit(self, kind: str, info: dict) -> None:
        if self.cancel_requested and helpers.is_cancellable_event(kind):
            raise InstallCancelled("Cancelled by user")
        self.q.put((kind, info))

    def _poll(self) -> None:
        try:
            while True:
                kind, info = self.q.get_nowait()
                if kind == "download":
                    total = info["bytes_total"] or 1
                    self.bar["value"] = 100 * info["bytes_done"] / total
                    self.status_var.set(
                        f"Downloading volume {info['index']} of "
                        f"{info['count']}: {info['volume']}")
                    self.detail_var.set(
                        f"{helpers.format_bytes(info['bytes_done'])} / "
                        f"{helpers.format_bytes(total)}")
                elif kind == "extract":
                    self.bar["value"] = 100
                    self.status_var.set(
                        f"Extracting volume {info['index']} of "
                        f"{info['count']}: {info['volume']}")
                    self.detail_var.set("")
                elif kind == "done":
                    self._show_finish(ok=True)
                elif kind == "cancelled":
                    self._show_finish(
                        ok=False,
                        message="Installation cancelled. Downloaded parts "
                                "are kept and will be resumed on retry.")
                elif kind == "error":
                    message = info.get("message", "")
                    if "HTTP 404" in message or "Cannot reach" in message:
                        message += (
                            "\n\nThis is usually a network/proxy issue. If you "
                            "use Clash Verge, enter http://127.0.0.1:7897 in "
                            "the Advanced proxy field and retry."
                        )
                    self._show_finish(ok=False, message=message)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _cancel(self) -> None:
        self.cancel_requested = True
        if hasattr(self, "status_var"):
            self.status_var.set("Cancelling…")

    def _finish(self) -> None:
        if getattr(self, "opts", {}).get("launch"):
            try:
                helpers.launch_app(self.install_dir)
            except Exception:
                pass
        self.root.destroy()

    # -- misc ----------------------------------------------------------------

    @staticmethod
    def _free_bytes(path: Path) -> int:
        import installer_core
        return installer_core._free_bytes(path)

    def _pretty_total(self) -> str:
        return helpers.format_bytes(self.manifest.total_bytes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EHT Analytica Setup")
    parser.add_argument("--uninstall", action="store_true",
                        help="run as uninstaller (used by Uninstall.exe)")
    parser.add_argument("--install-dir", default=None,
                        help="target installation directory")
    parser.add_argument("--base-url", default=None,
                        help="download base URL (overrides the embedded default)")
    parser.add_argument("--proxy", default=None,
                        help="HTTP proxy for downloads, e.g. 127.0.0.1:7897 "
                             "(overrides the Windows system proxy)")
    parser.add_argument("--manifest", default=None,
                        help="path to a manifest.json (dev mode)")
    args = parser.parse_args(argv)

    if args.uninstall:
        install_dir = Path(args.install_dir or helpers.default_install_dir())
        return run_uninstall(install_dir)

    if args.manifest:
        m = Manifest.from_dict(json.loads(
            Path(args.manifest).read_text(encoding="utf-8")))
    else:
        try:
            m = load_embedded_manifest()
        except SystemExit as e:
            # console=False: a traceback would be invisible, so show a dialog
            err = tk.Tk()
            err.withdraw()
            messagebox.showerror(WINDOW_TITLE, str(e))
            err.destroy()
            return 1

    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    WizardApp(root, m, args.base_url, args.install_dir, args.proxy)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
