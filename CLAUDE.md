# EHT Analytica

## Project Overview

Deep learning (U-Net) system for automatic recognition, tracking of EHT (Engineered Heart Tissue) pillars and calculation of contraction-relaxation metrics.
EHT is an in-vitro myocardial tissue assembled from PDMS pillars + cardiomyocytes + fibrin, capable of autonomous contraction-relaxation.

**Tech Stack**: Python (FastAPI) backend + Vue 3 (Vite) frontend + PyTorch GPU inference

## Quick Start

```bash
# First time setup
uv sync
# Double-click dev.bat, or manually:
uv run uvicorn backend.main:app --host 127.0.0.1 --port 9876 --reload
cd frontend && npm run dev
# Open http://localhost:5173
```

Note: `uv sync` installs Python dependencies. For the frontend, `dev.bat` runs `npm install` automatically on first launch.

## Coding Rules

- No Chinese (or any non-English) text in project files. All comments, docstrings, labels, and documentation must be written in English.

## Project Structure

```
EHT_Analytica/
├── backend/                    # FastAPI backend
│   ├── main.py                 # App entry point, CORS + static file mount
│   ├── launcher.py             # Desktop launcher (pywebview/browser)
│   ├── api/
│   │   ├── track.py            # Video open, model load (by name from models.json), tracking start
│   │   ├── analyze.py          # CSV read/validate/analyze/save
│   │   └── system.py           # Health check, file dialog, device info, job config
│   ├── models/schemas.py       # Pydantic request/response models
│   ├── services/
│   │   ├── video_service.py    # Video session management (metadata + first frame, single cap.open)
│   │   ├── tracking_service.py # Model loading + tracking (direct in-process inference, target_size propagation)
│   │   ├── analysis_service.py # Analysis pipeline (LengthDataAnalyzer wrapper)
│   │   ├── job_config.py       # Job config CRUD (<video-dir>/EHT-analytics/config.json)
│   │   └── task_manager.py     # Async task lifecycle + SSE progress push
│   └── utils/config.py         # Persistent app config (atomic writes via tempfile+os.replace)
├── frontend/                   # Vue 3 frontend (Vite)
│   └── src/
│       ├── App.vue             # Root component: Track/Analyze/Reports tabs, auto model load
│       ├── main.js             # Vue entry point
│       ├── components/
│       │   ├── track/          # TrackPanel, Sidebar (DEVICE badge), VideoCanvas, Progress
│       │   ├── analyze/        # AnalyzePanel (sample table + metadata + Force chart), Progress, CsvList
│       │   ├── reports/        # ReportsPanel (sample table + group label selector)
│       │   └── shared/         # FilePicker (path history memory), StatusIndicator
│       ├── composables/        # useApi (fetch wrapper), useSse, useTaskProgress
│       └── assets/main.css
├── functions/                  # Pure functions (no web dependencies)
│   ├── length_data_analyzer.py # length → force → peak detection → cycle segmentation → T80 → metrics
│   ├── tracker.py              # ROI tracking engine (frame loop + inference + interpolation + CSV output)
│   ├── roi.py                  # Pure data ROI class + color palette
│   └── video_processor.py      # OpenCV video metadata utilities
├── training/                    # Developer-only training & evaluation pipeline
│   ├── train_eval.ipynb         # Jupyter notebook: full training + eval workflow
│   ├── src/
│   │   ├── core/                # Training loop, losses, metrics, pipeline
│   │   └── dataset/             # Dataset, annotation tool, augmentations, inference collector
│   ├── configs/                 # Training hyperparameter configs
│   └── model/                   # Per-model training manifests
├── model/                       # Production inference models
│   ├── models.json              # Central model registry (name, display_name, weights, target_size)
├── model/unet_v3/               # U-Net v3 tracking model (single model, 256×256)
│   ├── unet_v3.py               # Model definition + load_model(weight_paths, device) -> EHTTracker
│   └── unet_v3_weights.pth      # Pure model weights (state_dict only, tracked in git)
├── config/                     # Persistent app config
│   └── app_config.json         # (gitignored) track_model_name, dir history, roi_names
├── build/                      # PyInstaller packaging scripts
├── dist/                       # PyInstaller build output (gitignored)
├── dev.bat                     # Dev mode launcher
├── requirements.txt
└── pyproject.toml
```

## Frontend Component Tree

```
App.vue
├─ TrackPanel.vue
│  ├─ TrackSidebar.vue          Device badge · Recording picker · Model dropdown · Track/Open buttons
│  ├─ VideoCanvas.vue           Canvas (ROI rectangle draw + name + delete) · resize responsive
│  └─ TrackProgressDialog.vue   SSE progress · error card
├─ AnalyzePanel.vue
│  ├─ FilePicker (Open)         Select EHT-analytics directory
│  ├─ Sample Table              Checkbox · Sample ID · Recording · ROI Name · Metadata columns · Length/Force/Metrics status
│  │  └─ Add Metadata button    Dynamically add editable column (key+value, built-in Recording/ROI Name non-editable/deletable)
│  ├─ Force Visualization       Chart.js line chart · auto-load force-status CSV on select
│  └─ AnalyzeProgressDialog     SSE progress · auto-save
└─ ReportsPanel.vue
   ├─ FilePicker (Open) + Refresh
   ├─ Sample Table              Checkbox · Sample ID · Recording · ROI Name · Metadata columns (only rows with metrics)
   └─ Group Label Selector      Recording · ROI Name · all metadata keys (multi-select, all unselected initially)
```

## API Endpoints

### System (`/api/system`)

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | `{"status": "ok"}` |
| GET | /device | GPU model+VRAM or CPU+RAM |
| GET | /models | List available models from models.json registry |
| GET | /config | Returns track_model_name, last_recording_dir, last_model_dir, roi_names |
| POST | /config/save-path | `{key, path}` Save directory history |
| POST | /config/save-roi-names | `{names: [...]}` Save ROI naming template |
| POST | /native-file-dialog | Native file/directory picker dialog (tkinter) |
| POST | /open-folder | Open directory/select file in file manager |
| POST | /list-csv | List all `.csv` in a directory |
| POST | /init-job-dir | Create `<video>/EHT-analytics/` + empty config.json |
| POST | /job-config | Read config.json (samples + metadata_keys) |
| POST | /save-metadata | Save metadata_keys + each sample's metadata values |

### Track (`/api/track`)

| Method | Path | Description |
|--------|------|-------------|
| POST | /video/open | Open video → return metadata + first frame base64 |
| POST | /model/load | Load model by name from models.json registry → background assemble EHTTracker (GPU preferred) |
| GET | /model/status | loaded, model_name, device, device_info |
| POST | /start | Pass video_id + ROIs + output_folder → background tracking |
| DELETE | /task/{id} | Cancel task |
| GET | /task/{id}/stream | SSE progress stream |

### Analyze (`/api/analyze`)

| Method | Path | Description |
|--------|------|-------------|
| POST | /read-csv | `{job_dir, paths}` → return columns + rows JSON |
| POST | /validate | Check CSV existence |
| POST | /start | Pass csv_paths → background analysis (pre-check time/length column validity) |
| POST | /save | Save force-status + metrics CSV to job directory |
| GET | /task/{id}/stream | SSE progress stream |

## Data Flow

```
Select Video ──(video/open)──→ metadata + first frame
     │
Draw ROI ──(frontend)──→ ROI coordinates
     │
Select model from dropdown ──(model/load)──→ load model directly in-process (GPU preferred)
     │
Click Track ──(track/start)──→ process_tracking()
     │                              ├─ frame-by-frame cap.read → direct in-process model inference
     │                              ├─ compute two-point distance = length
     │                              ├─ (if save_inferences) save ROI images + coords to inferences/
     │                              ├─ (if save_tracked_video) compose annotated tracked_video/
     │                              ├─ interpolate to uniform time grid (dt = min(t_diff))
     │                              └─ write lengths/{id}_{recording}_{roi}_length.csv
     │
Click Analyze ──(analyze/start)──→ LengthDataAnalyzer.analyze_lt_csvs()
     │                                   ├─ load length CSV
     │                                   ├─ unify_interval (uniform time step)
     │                                   ├─ calc_features (v_left, v_right, v, a, l_dev)
     │                                   ├─ calc_force (length → force, PDMS mechanics model)
     │                                   ├─ identify_peaks + identify_move_on_peaks
     │                                   ├─ seg_cycles + compute t80
     │                                   └─ return force_status DataFrames + metrics
     │
Click Save (auto) ──(analyze/save)──→ write:
          │    force-status/{id}_{recording}_{roi}_force_status.csv (time, force, status)
          │    metrics/{id}_{recording}_{roi}_metrics.csv (7 metric columns)
          │    update force_status_csv, metrics_csv fields in config.json
```

## Configuration Files

### model/models.json (model registry, tracked in git)
```json
{
  "models": [{
    "name": "unet_v3",
    "display_name": "EHT Tracker v3",
    "description": "...",
    "definition": "unet_v3/unet_v3.py",
    "weights": ["unet_v3/unet_v3_weights.pth"],
    "load_function": "load_model",
    "module": "unet_v3",
    "target_size": 256
  }]
}
```
Key fields: `name` (internal ID), `definition` (path to model .py), `weights` (paths to .pth files, relative to model/), `target_size` (input resolution, propagated to tracking metadata and inference collection).

### config/app_config.json (application-level, gitignored)
```json
{
  "track_model_name": "unet_v3",
  "last_recording_dir": "D:/.../data/HMBS-EHT/250904_rEHT",
  "last_model_dir": "D:/.../model/unet_v3",
  "roi_names": ["EHT-1", "EHT-2"]
}
```

### `<video-dir>/EHT-analytics/config.json` (job-level)
```json
{
  "samples": [{
    "id": "00000",
    "recording_name": "d14",
    "roi_name": "EHT-1",
    "length_csv": "lengths/00000_d14_EHT-1_length.csv",
    "force_status_csv": "force-status/00000_d14_EHT-1_force_status.csv",
    "metrics_csv": "metrics/00000_d14_EHT-1_metrics.csv",
    "metadata": {"Condition": "Control"}
  }],
  "metadata_keys": ["Condition"]
}
```

## Job Output Directory Structure

```
<video-dir>/EHT-analytics/
├── config.json
├── lengths/
│   └── {sample_id}_{recording}_{roi}_length.csv     (time, length)
├── force-status/
│   └── {sample_id}_{recording}_{roi}_force_status.csv  (time, force, status)
├── metrics/
│   └── {sample_id}_{recording}_{roi}_metrics.csv        (7 metric columns)
├── tracked_video/
│   └── {recording}_tracked.mp4                          (annotated tracking video)
└── inferences/
    ├── trainsets.json                                   (sample registry)
    └── {sample_id}_ROI.jpg                              (ROI image per sampled frame)
```

## Metric Columns

| Column | Meaning |
|--------|---------|
| EHT name | Composite identifier |
| Contraction Force | Peak contraction force (N) |
| Relaxation Force | Relaxation phase force (N) |
| Diastolic Tension | Diastolic tension (N) |
| Frequency | Contraction frequency (bpm) |
| Time to Peak 80% | T80 contraction (s) |
| Relaxation Time 80% | T80 relaxation (s) |

## Naming Conventions

- Sample ID: 5-digit zero-padded integer (00000, 00001...)
- Recording name: video filename without extension
- ROI name: user-defined, default "EHT-{n}"
- CSV naming: `{sample_id}_{recording}_{roi_name}_{type}.csv`
- Job directory: always `EHT-analytics/` co-located with the video

## Launcher & Heartbeat

`backend/launcher.py` starts uvicorn in a daemon thread, then opens the frontend:
1. **pywebview** (native window) — tried first; falls back to system browser on `ImportError` or runtime failure.
2. **System browser** (`_fallback_browser`) — opens `http://127.0.0.1:9876`, then runs a watchdog loop that exits the process when the browser tab is closed.

`backend/services/heartbeat.py` provides the close-detection mechanism:
- `HeartbeatMonitor` starts **dead** (`_last_beat = _UNSET = 0.0`); `is_alive()` returns `False` until the first `beat()`.
- The frontend pings `/api/system/heartbeat` every 3 s.
- The launcher calls `beat()` after an 8 s grace period to activate the monitor.
- If no beat arrives within 5 s, the watchdog calls `sys.exit(0)`.

## Release Build

Build the self-contained release package:

```bash
uv run python build/build.py
```

### Build pipeline (3 stages)

1. **Frontend build** — `npm run build` outputs to `frontend/dist/` (includes `public/` static assets like `logo.ico`)
2. **PyInstaller** — packages `backend/`, `frontend/dist/`, `functions/`, `training/` into `dist/EHT_Analytica/` with a single `.exe` entry point
3. **Model assembly** — copies model definitions + weights into `dist/EHT_Analytica/model/` based on `model/models.json` registry

### Output structure

```
dist/EHT_Analytica/
├── EHT_Analytica.exe          # Entry point (icon embedded from img/logo.ico)
├── model/
│   ├── models.json            # Model registry (copied from source)
│   └── <module>/              # Per-model .py + .pth files
├── frontend_dist/             # Built Vue frontend (served as static files)
├── functions/                 # Pure Python functions
├── backend/                   # FastAPI backend
└── _internal/                 # PyInstaller runtime
```

### Model exclusion

`build/build.py` packages every model listed in `model/models.json` whose weight files exist. To keep a model out of release builds, remove its entry from the registry (or delete its weights so the assembly step skips it).

### Release checklist

- `.exe` icon: `img/logo.ico` if present (gitignored), else `frontend/public/logo.ico` (tracked — used by CI)
- `model/unet_v3/unet_v3_weights.pth` must exist (tracked in git)
- Run `build/build.py` from the project root with the uv-managed venv active

## Development Environment

- **Python**: uv-managed venv (`.venv/`), PyTorch 2.x+cu126
- **GPU**: NVIDIA GeForce RTX 4070 Ti SUPER (16376 MB)
- **Node.js**: npm, Vite 6
- **Ports**: backend 9876, frontend dev 5173 (Vite proxies /api → 9876)
