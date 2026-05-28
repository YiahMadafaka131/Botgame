"""Recover tap locations from the "show touches" overlay in a recorded video.

Fallback for when you only have a screen recording (no getevent log): if you
enabled Developer options -> "Show taps", Android draws a bright circle at each
touch point. We find the brightest blob per frame and return its centroid.

Less reliable than getevent (bright game art can fool it, and it cannot see
multitouch), so prefer getevent when available. numpy-only, no OpenCV needed.
"""

from __future__ import annotations

import numpy as np


def detect_touch(
    frame_rgb: np.ndarray,
    *,
    brightness_threshold: int = 230,
    min_pixels: int = 30,
) -> tuple[int, int] | None:
    """Return (x, y) of the touch overlay, or None if no clear blob is found.

    `frame_rgb` is an (H, W, 3) uint8 array. We threshold on luminance and take
    the centroid of the bright pixels.
    """
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError("expected an (H, W, 3) RGB array")

    lum = frame_rgb.mean(axis=2)
    mask = lum >= brightness_threshold
    count = int(mask.sum())
    if count < min_pixels:
        return None

    ys, xs = np.nonzero(mask)
    return int(round(xs.mean())), int(round(ys.mean()))
