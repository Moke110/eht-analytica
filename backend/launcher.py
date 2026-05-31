"""Desktop launcher for EHT Analytica.

Starts the FastAPI backend and opens the frontend in a browser or native webview.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path


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


def main():
    # Ensure the project root is on sys.path so `functions.*` imports work
    base = _get_base_dir()
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))

    # Check frontend exists
    frontend = _find_frontend()
    if frontend is None:
        print("WARNING: No built frontend found. API will work, but open /docs for testing.")

    host = "127.0.0.1"
    port = 9876

    # Start uvicorn in a daemon thread
    import uvicorn
    from backend.main import app

    def serve():
        uvicorn.run(app, host=host, port=port, log_level="info")

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    time.sleep(1.5)

    url = f"http://{host}:{port}"

    # Try pywebview first, fall back to browser
    try:
        import webview
        webview.create_window("EHT Analytica", url, width=1280, height=900)
        webview.start()
    except ImportError:
        import webbrowser
        print(f"Starting EHT Analytica at {url}")
        webbrowser.open(url)
        print("Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down.")


if __name__ == "__main__":
    main()
