"""FastAPI application — serves the Vue frontend and API routes."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.system import router as system_router
from backend.api.track import router as track_router
from backend.api.analyze import router as analyze_router


def _get_frontend_dist() -> Path | None:
    """Locate the built Vue frontend directory."""
    if getattr(sys, 'frozen', False):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parent.parent

    candidates = [
        base / "frontend_dist",
        base / "frontend" / "dist",
    ]
    for c in candidates:
        if (c / "index.html").exists():
            return c
    return None


def create_app() -> FastAPI:
    app = FastAPI(title="EHT Analytica", version="0.2.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routers
    app.include_router(system_router)
    app.include_router(track_router)
    app.include_router(analyze_router)

    # Serve Vue frontend as static files (production mode)
    frontend_dir = _get_frontend_dist()
    if frontend_dir:
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app


app = create_app()
