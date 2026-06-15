"""Desktop launcher for EHT Analytica.

Starts the FastAPI backend and opens the frontend in a browser or native webview.
Exits automatically when the browser tab/window is closed (heartbeat watchdog).
"""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from pathlib import Path


def _setup_stdio() -> None:
    """Redirect stdout/stderr to a log file when running frozen + GUI mode.

    In PyInstaller ``console=False`` builds, ``sys.stdout`` and
    ``sys.stderr`` are ``None`` — any ``print()`` or logging output
    raises an exception.  Redirect to a log file so startup errors
    are captured and uvicorn logging works.
    """
    if not getattr(sys, 'frozen', False):
        return
    if sys.stdout is not None and sys.stderr is not None:
        return

    log_dir = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path.cwd()
    log_path = log_dir / 'eht_analytica.log'

    # Truncate on each launch to avoid unbounded growth
    f = open(str(log_path), 'w', encoding='utf-8')
    sys.stdout = f
    sys.stderr = f
    print(f"EHT Analytica log — {time.strftime('%Y-%m-%d %H:%M:%S')}")


def _get_base_dir() -> Path:
    """Get the application base directory (dev or PyInstaller)."""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


def _find_frontend() -> Path | None:
    base = _get_base_dir()
    candidates = [
        base / "frontend_dist",
        base / "frontend" / "dist",
    ]
    for c in candidates:
        if (c / "index.html").exists():
            return c
    return None


def _wait_for_server(url: str, timeout: float = 15.0) -> bool:
    """Poll the health endpoint until the server is ready or timeout expires."""
    import urllib.request
    health_url = f"{url}/api/system/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def _fallback_browser(url: str) -> None:
    """Fall back to system browser with heartbeat watchdog."""
    import webbrowser
    from backend.services.heartbeat import heartbeat_monitor

    print(f"Starting EHT Analytica at {url}")
    webbrowser.open(url)

    # Watchdog: exit when browser disconnects (heartbeat stops)
    time.sleep(8)  # grace period for browser to load and start pinging
    heartbeat_monitor.beat()  # prime: treat end-of-grace as a fresh beat
    while heartbeat_monitor.is_alive():
        time.sleep(1)
    print("Browser disconnected. Shutting down EHT Analytica.")
    sys.exit(0)


def main():
    _setup_stdio()

    try:
        _main_impl()
    except Exception:
        traceback.print_exc()
        # Keep the console/log visible long enough to read the error
        print("\nFATAL: Launcher crashed. See traceback above.")
        time.sleep(30)
        sys.exit(1)


def _main_impl():
    # Ensure the project root is on sys.path so `functions.*` imports work.
    # When frozen, also add the EXE directory so the external model/ is importable.
    base = _get_base_dir()
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))

    if getattr(sys, 'frozen', False):
        exe_dir = str(Path(sys.executable).parent)
        if exe_dir not in sys.path:
            sys.path.insert(0, exe_dir)

    # Check frontend exists
    frontend = _find_frontend()
    if frontend is None:
        print("WARNING: No built frontend found. API will work, but open /docs for testing.")

    host = "127.0.0.1"
    port = 9876

    # Start uvicorn in a daemon thread
    import uvicorn
    from backend.main import app

    # Capture uvicorn startup errors from the daemon thread
    serve_error = []

    def serve():
        try:
            uvicorn.run(app, host=host, port=port, log_level="info")
        except Exception as e:
            serve_error.append(e)
            traceback.print_exc()

    t = threading.Thread(target=serve, daemon=True)
    t.start()

    url = f"http://{host}:{port}"

    # Wait for server to be ready (with health-check, not blind sleep)
    print("Waiting for server to start...")
    if not _wait_for_server(url, timeout=15.0):
        if serve_error:
            print(f"ERROR: Server failed to start: {serve_error[0]}")
        else:
            print("ERROR: Server did not become ready within 15 seconds.")
        print("Check eht_analytica.log for details.")
        time.sleep(10)
        sys.exit(1)

    print("Server is ready.")

    # Check if serve thread died
    if serve_error:
        print(f"ERROR: Server thread died: {serve_error[0]}")
        time.sleep(10)
        sys.exit(1)

    # Try pywebview first, fall back to browser
    try:
        import webview
    except ImportError:
        webview = None

    if webview is not None:
        try:
            webview.create_window("EHT Analytica", url, width=1280, height=900)
            webview.start()
        except Exception:
            traceback.print_exc()
            print("pywebview failed at runtime, falling back to browser...")
            _fallback_browser(url)
    else:
        _fallback_browser(url)


if __name__ == "__main__":
    main()
