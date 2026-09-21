# EHT Analytica

A deep learning EHT (Engineered Heart Tissue) contraction analysis application.
Uses a U-Net model for automatic pillar detection and tracking, with a dropdown
selector to switch between available models at runtime.

**Tech**: Python (FastAPI) backend + Vue 3 (Vite) frontend + PyTorch GPU inference.

## Quick Start

```bash
# First time setup
uv sync
cd frontend && npm install

# Dev mode
dev.bat
# Or manually:
uv run uvicorn backend.main:app --host 127.0.0.1 --port 9876 --reload
cd frontend && npm run dev
# Open http://localhost:5173
```

## Features

- **Track**: Open video → draw ROIs → select model → track. Outputs per-ROI length CSVs.
- **Analyze**: Load length CSVs → compute force → peak detection → cycle metrics (T80, frequency, etc.).
- **Reports**: View metrics across samples with grouping and visualization.
- **Save Inferences**: Optional — saves ROI images + model predictions to `EHT-analytics/inferences/` for training data review.
- **Tracked Video**: Optional — composes an annotated video with overlaid ROI cells and crosshair overlays.

## Requirements

- Python 3.10+ (managed by uv)
- Node.js
- NVIDIA GPU (optional, CPU fallback)
- `model/unet_v3/unet_v3_weights.pth` — default model weights (tracked in git)
- `model/unet_v2/` — optional legacy model (gitignored, see [Models](#models))

## Models

Select a tracking model from the dropdown in the Track tab. The application
discovers available models automatically from `model/models.json`.

| Model | Name | Description |
|-------|------|-------------|
| **EHT Tracker v3** | `unet_v3` | Single U-Net (256×256, 3-stage). Fast, lightweight. Default and recommended. Weights are tracked in git. |
| **EHT Tracker v2** | `unet_v2` | 5-model ensemble (512×512, 4-stage). Superseded by v3. Weights are gitignored (large files). |

### Adding a New Model

1. Place model definition + weights in `model/{name}/` following the naming convention
2. Add an entry to `model/models.json`
3. The model appears automatically in the Track tab dropdown

### Model Directories

Each model directory under `model/` contains:
- `{name}.py` — Model architecture + `load_model(weight_paths, device)` function
- `{name}_weights.pth` — Pure model weights (state_dict). v3 weights are tracked in git; v2 weights are gitignored (use `model/unet_v2/` locally for v2).

## Developer Docs

- `CLAUDE.md` — Full project reference (structure, API endpoints, data flow, config)
- `training/README.md` — Training pipeline setup, annotation tool, adding new models
- `training/CLAUDE.md` — Training codebase reference for LLM context

## Packaging

```bash
python build/build.py        # PyInstaller build → dist/
```

The build script reads `model/models.json` and includes all models whose weight files exist — it is model-agnostic rather than hardcoded to a specific version.

## Download

Prebuilt packages for Windows and macOS (Apple Silicon) are available on the
[Releases](https://github.com/Moke110/eht_analytica/releases) page. Release builds
are CPU-only; build from source for NVIDIA GPU support.

## License

Released under the [MIT License](LICENSE).
