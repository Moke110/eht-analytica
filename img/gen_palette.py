"""Generate color palette preview PNG for Reports plot."""
from __future__ import annotations

import colorsys
import math
from PIL import Image, ImageDraw, ImageFont

# Edge colors (original palette)
EDGE_HEX = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]

NAMES = [
    "Blue", "Orange", "Green", "Red", "Purple",
    "Brown", "Magenta", "Gray", "Gold", "Teal",
]


def hex_to_hsv(hx: str) -> tuple[float, float, float]:
    r, g, b = int(hx[1:3], 16) / 255, int(hx[3:5], 16) / 255, int(hx[5:7], 16) / 255
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360, s * 100, v * 100


def hsv_to_hex(h: float, s: float, v: float) -> str:
    r, g, b = colorsys.hsv_to_rgb(h / 360, s / 100, v / 100)
    return f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"


def derive(hex_color: str) -> tuple[str, str, str]:
    """Return (edge, fill, dot) hex triple."""
    h, s, v = hex_to_hsv(hex_color)
    fill = hsv_to_hex(h, s * 0.40, min(v * 1.30, 100))
    dot = hsv_to_hex(h, s * 0.65, min(v * 1.15, 100))
    return hex_color, fill, dot


def hsv_str(hx: str) -> str:
    h, s, v = hex_to_hsv(hx)
    # Gray: H is meaningless
    if s < 0.5:
        return f" —    {s:3.0f}%  {v:3.0f}%"
    return f"{h:3.0f}°  {s:3.0f}%  {v:3.0f}%"


# Layout constants
SWATCH_W = 90
ROW_H = 56
HEADER_H = 52
LABEL_W = 80
GROUP_W = 72
COL_GAP = 8
PAD = 20

COLS = 10
ROLES = 3  # Edge, Fill, Dot
ROLE_NAMES = ["Edge\n(bar border)", "Fill\n(bar bg)", "Dot\n(sample pt)"]

TOTAL_W = LABEL_W + COLS * (SWATCH_W + COL_GAP) * ROLES + COL_GAP
TOTAL_H = HEADER_H + ROLES * 10 + ROW_H * ROLES + PAD * 2

img = Image.new("RGB", (TOTAL_W, TOTAL_H), "#F8F8F8")
draw = ImageDraw.Draw(img)

# Try to load a font, fallback to default
try:
    font_sm = ImageFont.truetype("consola.ttf", 13)
    font_md = ImageFont.truetype("consola.ttf", 15)
    font_bold = ImageFont.truetype("consolab.ttf", 16)
except Exception:
    font_sm = ImageFont.load_default()
    font_md = ImageFont.load_default()
    font_bold = ImageFont.load_default()

# Title
draw.text((PAD, 14), "Reports Plot — Color Palette", fill="#222", font=font_bold)

y_top = HEADER_H

# Draw column headers (group names) and sub-headers (role names)
for ci in range(COLS):
    x_base = LABEL_W + ci * (SWATCH_W + COL_GAP) * ROLES
    tx = x_base + (SWATCH_W * ROLES + COL_GAP * (ROLES - 1)) // 2
    draw.text((tx - 30, HEADER_H + 2), NAMES[ci], fill="#222", font=font_md)

# Draw role labels
for ri in range(ROLES):
    ry = HEADER_H + 28 + ri * ROW_H + ROW_H // 2 - 8
    draw.text((4, ry), ROLE_NAMES[ri], fill="#444", font=font_sm)

# Draw swatches
for ci in range(COLS):
    edge, fill, dot = derive(EDGE_HEX[ci])
    colors = [edge, fill, dot]

    # Compute HSV for each role
    hsvs = [hex_to_hsv(c) for c in colors]

    for ri in range(ROLES):
        x_base = LABEL_W + ci * (SWATCH_W + COL_GAP) * ROLES + ri * (SWATCH_W + COL_GAP)
        ry = HEADER_H + 28 + ri * ROW_H

        # Color swatch
        draw.rectangle([x_base, ry, x_base + SWATCH_W, ry + ROW_H - 8],
                       fill=colors[ri], outline="#BBB", width=1)

        # Hex label
        hex_text = colors[ri]
        tx = x_base + 6
        ty = ry + ROW_H - 24
        draw.text((tx, ty), hex_text, fill="#222", font=font_sm)

        # HSV label
        hsv_label = hsv_str(colors[ri])
        draw.text((tx, ty + 16), hsv_label, fill="#555", font=font_sm)

out_path = "img/color_palette.png"
img.save(out_path)
print(f"Saved: {out_path}")
print(f"Size: {TOTAL_W}x{TOTAL_H}")
