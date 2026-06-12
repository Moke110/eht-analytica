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

## Requirements

- Python 3.10+ (managed by uv)
- Node.js
- NVIDIA GPU (optional, CPU fallback)
- Model weight files in `model/unet_v2/` and `model/unet_v3/` (see [Models](#models))

## Models

Select a tracking model from the dropdown in the Track tab. The application
discovers available models automatically from `model/models.json`.

| Model | Name | Description |
|-------|------|-------------|
| **EHT Tracker v3** | `unet_v3` | Single U-Net (256×256, 3-stage). Fast, lightweight. Recommended for most use cases. |
| **EHT Tracker v2** | `unet_v2` | 5-model ensemble (512×512, 4-stage). Higher accuracy, ~2× slower inference. |

### Adding a New Model

1. Place model definition + weights in `model/{name}/` following the naming convention
2. Add an entry to `model/models.json`
3. The model appears automatically in the Track tab dropdown

### Model Directories

Each model directory under `model/` contains:
- `{name}.py` — Model architecture + `load_model(weight_paths, device)` function
- `{name}_weights.pth` — Pure model weights (state_dict, no training metadata)
