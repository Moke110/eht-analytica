# EHT Analytica

A deep learning EHT (Engineered Heart Tissue) contraction analysis application.

**Tech**: Python (FastAPI) backend + Vue 3 (Vite) frontend.

## Quick Start

```bash
# Dev mode
dev.bat
# Or manually:
python -m uvicorn backend.main:app --host 127.0.0.1 --port 9876 --reload
cd frontend && npm run dev
# Open http://localhost:5173
```

## Requirements

- Python 3.10+ with conda or venv
- Node.js
- NVIDIA GPU (optional, CPU fallback)
