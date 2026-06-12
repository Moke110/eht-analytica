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
│   │   ├── tracking_service.py # Model loading + tracking execution (direct in-process inference)
│   │   ├── analysis_service.py # Analysis pipeline (LengthDataAnalyzer wrapper)
│   │   ├── job_config.py       # Job config CRUD (<video-dir>/EHT-analytics/config.json)
│   │   └── task_manager.py     # Async task lifecycle + SSE progress push
│   └── utils/config.py         # Persistent app config (track_model_name, dir history, roi_names)
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
├── model/                       # Production inference models
│   ├── models.json              # Central model registry (name, display_name, weights, classes)
├── model/unet_v2/               # U-Net v2 tracking model (5-model ensemble, 512×512)
│   ├── unet_v2.py               # Model definition + load_model(weight_paths, device) -> EHTTracker
│   └── unet_v2_R*_weights.pth   # 5 pure weight files (state_dict only)
├── model/unet_v3/               # U-Net v3 tracking model (single model, 256×256)
│   ├── unet_v3.py               # Model definition + load_model(weight_paths, device) -> EHTTracker
│   └── unet_v3_weights.pth      # Pure model weights (state_dict only)
├── config/                     # Persistent app config
│   └── app_config.json         # (gitignored) track_model_name, dir history, roi_names
├── build/                      # PyInstaller packaging scripts
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
└── metrics/
    └── {sample_id}_{recording}_{roi}_metrics.csv        (7 metric columns)
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

## Development Environment

- **Python**: uv-managed venv (`.venv/`), PyTorch 2.x+cu126
- **GPU**: NVIDIA GeForce RTX 4070 Ti SUPER (16376 MB)
- **Node.js**: npm, Vite 6
- **Ports**: backend 9876, frontend dev 5173 (Vite proxies /api → 9876)
