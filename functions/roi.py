"""Pure-data ROI class and color palette (no wxPython dependency)."""

from dataclasses import dataclass


# Predefined bright colors for ROIs (RGB tuples)
COLORS: list[tuple[int, int, int]] = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 128, 255),    # Light Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (255, 128, 255),  # Light Purple
    (128, 255, 128),  # Light Green
    (255, 192, 203),  # Pink
]


@dataclass
class Rect:
    x: int
    y: int
    width: int
    height: int


class ROI:
    """Represents a single Region of Interest (no wxPython dependency)."""

    def __init__(self, rect, color, name, original_frame_size, canvas_size):
        self.rect = rect  # Rect in canvas coordinates
        self.color = color  # (r, g, b) tuple
        self.name = name
        self.original_frame_size = original_frame_size  # (w, h)
        self.canvas_size = canvas_size  # (w, h)
        self.pixel_rect = self._convert_to_pixel_space()

    def _convert_to_pixel_space(self):
        """Convert canvas coordinates to original frame pixel coordinates."""
        orig_width, orig_height = self.original_frame_size
        canvas_width, canvas_height = self.canvas_size

        scale = min(canvas_width / orig_width, canvas_height / orig_height)

        scaled_width = int(orig_width * scale)
        scaled_height = int(orig_height * scale)
        offset_x = (canvas_width - scaled_width) // 2
        offset_y = (canvas_height - scaled_height) // 2

        x = self.rect.x - offset_x
        y = self.rect.y - offset_y
        w = self.rect.width
        h = self.rect.height

        pixel_x = int(x / scale)
        pixel_y = int(y / scale)
        pixel_w = int(w / scale)
        pixel_h = int(h / scale)

        pixel_x = max(0, min(pixel_x, orig_width))
        pixel_y = max(0, min(pixel_y, orig_height))
        pixel_w = max(0, min(pixel_w, orig_width - pixel_x))
        pixel_h = max(0, min(pixel_h, orig_height - pixel_y))

        return Rect(pixel_x, pixel_y, pixel_w, pixel_h)

    def get_pixel_coordinates(self) -> dict:
        """Get ROI coordinates in original frame pixel space."""
        return {
            'name': self.name,
            'x': self.pixel_rect.x,
            'y': self.pixel_rect.y,
            'width': self.pixel_rect.width,
            'height': self.pixel_rect.height,
            'color': self.color,
        }
