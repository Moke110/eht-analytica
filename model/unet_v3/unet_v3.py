"""
EHT tracking model v3 — single U-Net (256×256) for pillar coordinate prediction.

Design:
- Single model trained once
- 256×256 input/output
- 3 encoder/decoder stages
- Bottleneck channels: 256
- ~70% fewer parameters, ~2× faster inference

This module runs in eager mode (no TorchScript) for CUDA compatibility.
"""
import os
from typing import Tuple

import torch
import torch.nn as nn


class CircleCenterNet(nn.Module):
    """U-Net (3-stage) that predicts a heatmap of pillar centers from a grayscale ROI.

    Input:  (B, 1, 256, 256)  grayscale ROI crop
    Output: (B, 1, 256, 256)  sigmoid heatmap (pillar center likelihood)
    """

    def __init__(self, input_channels=1, output_size=(256, 256)):
        super().__init__()
        self.output_size = output_size

        # ── Encoder (3 pooling stages: 256→128→64→32) ────
        self.enc1_1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.enc1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)         # 256→128

        self.enc2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)         # 128→64

        self.enc3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)         # 64→32

        # ── Bottleneck (32×32, 256 channels) ──────────────
        self.bottom1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bottom2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn_bottom1 = nn.BatchNorm2d(256)
        self.bn_bottom2 = nn.BatchNorm2d(256)

        # ── Decoder (3 upsampling stages) with skip ────────
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 32→64
        self.dec3_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)       # 128+128=256
        self.dec3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn_up3_1 = nn.BatchNorm2d(128)
        self.bn_up3_2 = nn.BatchNorm2d(128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # 64→128
        self.dec2_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)        # 64+64=128
        self.dec2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_up2_1 = nn.BatchNorm2d(64)
        self.bn_up2_2 = nn.BatchNorm2d(64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)    # 128→256
        self.dec1_1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)         # 32+32=64
        self.dec1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn_up1_1 = nn.BatchNorm2d(32)
        self.bn_up1_2 = nn.BatchNorm2d(32)

        # ── Output ─────────────────────────────────────────
        self.output = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Encoder
        e1 = self.relu(self.bn1_1(self.enc1_1(x)))
        e1 = self.relu(self.bn1_2(self.enc1_2(e1)))
        p1 = self.pool1(e1)

        e2 = self.relu(self.bn2_1(self.enc2_1(p1)))
        e2 = self.relu(self.bn2_2(self.enc2_2(e2)))
        p2 = self.pool2(e2)

        e3 = self.relu(self.bn3_1(self.enc3_1(p2)))
        e3 = self.relu(self.bn3_2(self.enc3_2(e3)))
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.relu(self.bn_bottom1(self.bottom1(p3)))
        b = self.relu(self.bn_bottom2(self.bottom2(b)))

        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.relu(self.bn_up3_1(self.dec3_1(d3)))
        d3 = self.relu(self.bn_up3_2(self.dec3_2(d3)))

        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.relu(self.bn_up2_1(self.dec2_1(d2)))
        d2 = self.relu(self.bn_up2_2(self.dec2_2(d2)))

        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.relu(self.bn_up1_1(self.dec1_1(d1)))
        d1 = self.relu(self.bn_up1_2(self.dec1_2(d1)))

        return self.sigmoid(self.output(d1))


class EHTTracker(nn.Module):
    """Single CircleCenterNet model wrapper for end-to-end pillar tracking.

    Handles pre-processing (pad→resize→normalize), model inference, and
    post-processing (heatmap→coords→original pixel space).
    """

    def __init__(self, models: nn.ModuleList, device: torch.device):
        super().__init__()
        models.to(device)
        self.models = models
        self.num_models = len(models)
        self.target_size = 256
        self.num_peaks = 2
        self.blob_sd = 6.0                           # gaussian heatmap sigma for 256×256
        self.exclusion_radius = 1.96 * self.blob_sd  # ≈ 11.76 px

    def pre_process(self, image: torch.Tensor) -> Tuple[torch.Tensor, int, int, int, int, int]:
        orig_h, orig_w = image.shape[0], image.shape[1]
        S = max(orig_w, orig_h)

        canvas = image.new_zeros((S, S), dtype=torch.float32)
        dx = (S - orig_w) // 2
        dy = (S - orig_h) // 2
        canvas[dy:dy + orig_h, dx:dx + orig_w] = image

        canvas_4d = canvas.unsqueeze(0).unsqueeze(0)
        resized = torch.nn.functional.interpolate(
            canvas_4d, size=(self.target_size, self.target_size),
            mode='bilinear', align_corners=False,
        )
        normalized = resized / 255.0
        return normalized, orig_w, orig_h, dx, dy, S

    def _single_forward(self, preprocessed: torch.Tensor) -> torch.Tensor:
        """Run the single model and return its output."""
        with torch.no_grad():
            return self.models[0](preprocessed)

    def post_process(self, heatmap: torch.Tensor,
                     orig_w: int, orig_h: int,
                     dx: int, dy: int, S: int) -> torch.Tensor:
        coords_norm = self._heatmap2coords(heatmap)
        coords_orig = torch.zeros_like(coords_norm)
        target_f = float(self.target_size)
        S_f = float(S)

        for i in range(self.num_peaks):
            x_target = coords_norm[i, 0] * target_f
            y_target = coords_norm[i, 1] * target_f
            x_canvas = x_target * (S_f / target_f)
            y_canvas = y_target * (S_f / target_f)
            coords_orig[i, 0] = x_canvas - float(dx)
            coords_orig[i, 1] = y_canvas - float(dy)

        return coords_orig

    def _heatmap2coords(self, heatmap: torch.Tensor) -> torch.Tensor:
        h_map = heatmap[0, 0].clone()
        coords = torch.full((self.num_peaks, 2), float('nan'), dtype=torch.float32,
                            device=h_map.device)
        working = h_map.clone()

        for i in range(self.num_peaks):
            max_val = working.max()
            if max_val.item() == 0.0:
                break
            max_idx = working.reshape(-1).argmax()
            y_idx = (max_idx // self.target_size).to(torch.float32)
            x_idx = (max_idx % self.target_size).to(torch.float32)

            coords[i, 0] = x_idx / float(self.target_size)
            coords[i, 1] = y_idx / float(self.target_size)

            yg = torch.arange(self.target_size, dtype=torch.float32,
                              device=h_map.device).unsqueeze(1)
            xg = torch.arange(self.target_size, dtype=torch.float32,
                              device=h_map.device).unsqueeze(0)
            dist = torch.sqrt((xg - x_idx) ** 2 + (yg - y_idx) ** 2)
            working = working * (dist > self.exclusion_radius).float()

        return coords

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        preprocessed, ow, oh, dx, dy, S = self.pre_process(image)
        heatmap = self._single_forward(preprocessed)
        return self.post_process(heatmap, ow, oh, dx, dy, S)

    def forward_with_heatmap(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Like forward(), but also returns the model heatmap.

        Returns
        -------
        coords : torch.Tensor  shape (num_peaks, 2) in original pixel coordinates
        heatmap : torch.Tensor  shape (1, 1, target_size, target_size) sigmoid [0, 1]
        """
        preprocessed, ow, oh, dx, dy, S = self.pre_process(image)
        heatmap = self._single_forward(preprocessed)
        coords = self.post_process(heatmap, ow, oh, dx, dy, S)
        return coords, heatmap


def load_model(weight_paths: list[str], device: torch.device) -> EHTTracker:
    """Load model weights from .pth files and return an assembled EHTTracker.

    Each .pth file should contain a raw model state_dict (OrderedDict).
    For backward compatibility, training checkpoints containing a
    ``"model_state_dict"`` key are also handled.

    Parameters
    ----------
    weight_paths : list[str]
        Absolute paths to one or more .pth weight files.
        For v3, typically a single file.
    device : torch.device
        Target device (cuda or cpu).

    Returns
    -------
    EHTTracker
        Fully assembled model on the target device, in eval mode.
    """
    models = nn.ModuleList()
    for pth_path in weight_paths:
        if not os.path.isfile(pth_path):
            raise FileNotFoundError(f"Weight file not found: {pth_path}")
        ckpt = torch.load(pth_path, map_location=device, weights_only=False)
        # Handle both training checkpoints and raw state_dicts
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        m = CircleCenterNet()
        m.load_state_dict(state_dict)
        m.to(device)
        m.eval()
        models.append(m)

    return EHTTracker(models, device)
