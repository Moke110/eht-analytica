import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt


def select_device():
    """Return torch.device: prefer CUDA if available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image_as_tensor(path: Path, device: torch.device) -> torch.Tensor:
    """Load image as grayscale float32 tensor [H, W] with values 0-255 on device."""
    img = Image.open(path).convert("L")
    arr = np.asarray(img).astype(np.float32)
    tensor = torch.from_numpy(arr)
    return tensor.to(device)


def load_gt_coords(path: Path) -> np.ndarray:
    """Load GT coordinates CSV with columns x,y and return as Nx2 numpy array."""
    df = pd.read_csv(path)
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError(f"GT csv missing x,y columns: {path}")
    coords = df[["x", "y"]].to_numpy(dtype=np.float32)
    return coords


def pairwise_min_distances(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """For each predicted point, compute distance to nearest GT point."""
    if gts.size == 0:
        return np.full((preds.shape[0],), np.nan)
    dists = np.sqrt(((preds[:, None, :] - gts[None, :, :]) ** 2).sum(axis=2))
    return dists.min(axis=1)


def main():
    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parents[2]

    # Paths
    model_path = repo_root / "model" / "track_unet_v2" / "EHTTracker.pt"
    test_set_dir = base_dir / "test_set"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Device
    device = select_device()
    print(f"Using device: {device}")

    # Load TorchScript model onto device
    print(f"Loading model from: {model_path}")
    # torch.jit.load supports map_location
    try:
        tracker = torch.jit.load(str(model_path), map_location=device)
    except TypeError:
        # Older torch versions expect map_location as function/string
        tracker = torch.jit.load(str(model_path))

    tracker.eval()

    # Collect PNG files
    pngs = sorted([p for p in test_set_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if len(pngs) < 9:
        raise RuntimeError(f"Not enough test images in {test_set_dir} (found {len(pngs)})")

    chosen = random.sample(pngs, 9)

    preds_all = []
    gts_all = []
    imgs_display = []

    for p in chosen:
        coords_csv = p.with_name(p.stem + "_coords.csv")

        img_tensor = load_image_as_tensor(p, device)

        # Forward (model expects [H, W] tensor with values 0-255)
        with torch.no_grad():
            out = tracker(img_tensor)

        # out is [num_peaks, 2] on device
        out_np = out.cpu().numpy().astype(np.float32)

        # Load GT
        if coords_csv.exists():
            gt = load_gt_coords(coords_csv)
        else:
            gt = np.zeros((0, 2), dtype=np.float32)

        # Store
        preds_all.append(out_np)
        gts_all.append(gt)

        # Save image for plotting (RGB)
        img = Image.open(p).convert("L")
        imgs_display.append(np.asarray(img))

    # Compute distances for error metric
    distances = []
    for preds, gts in zip(preds_all, gts_all):
        if preds is None or preds.size == 0:
            continue
        d = pairwise_min_distances(preds, gts)
        # Filter NaNs
        d = d[~np.isnan(d)]
        if d.size > 0:
            distances.extend(d.tolist())

    avg_error = float(np.mean(distances)) if distances else float('nan')

    # Plot 3x3 grid
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    axs = axs.flatten()

    for idx, ax in enumerate(axs):
        ax.imshow(imgs_display[idx], cmap='gray')
        ax.axis('off')

        preds = preds_all[idx]
        gts = gts_all[idx]

        # Plot GT as blue crosses
        if gts is not None and gts.size:
            ax.scatter(gts[:, 0], gts[:, 1], c='blue', marker='x', s=80, label='GT')

        # Plot preds as red crosses
        if preds is not None and preds.size:
            ax.scatter(preds[:, 0], preds[:, 1], c='red', marker='x', s=80, label='Pred')

    # Put average error text at bottom
    fig.suptitle('EHTTracker predictions (red) vs GT (blue)', fontsize=16)
    fig.text(0.5, 0.02, f'Average pixel error (pred -> nearest GT): {avg_error:.2f} px', ha='center', fontsize=12)

    out_file = base_dir / 'ehttracker_test_results.png'
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out_file, dpi=150)
    print(f"Saved result plot to: {out_file}")
    print(f"Average pixel error: {avg_error:.2f}")


if __name__ == '__main__':
    main()
