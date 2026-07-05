"""Screen capture over ADB.

Baseline uses `adb exec-out screencap -p`: dependency-free and reliable, but
only ~1-5 fps. For real-time play swap this for a scrcpy/minicap stream later;
the rest of the pipeline only needs `grab()` to return an RGB numpy array.
"""

from __future__ import annotations

import io
import os

import numpy as np
from PIL import Image

from .device import AdbDevice


class ScreenCapture:
    def __init__(self, device: AdbDevice):
        self.device = device

    def grab_png(self) -> bytes:
        """Raw PNG bytes of the current screen."""
        return self.device.exec_out("screencap", "-p")

    def grab(self) -> np.ndarray:
        """Current screen as an (H, W, 3) uint8 RGB array."""
        png = self.grab_png()
        img = Image.open(io.BytesIO(png)).convert("RGB")
        return np.asarray(img)

    def save(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(self.grab_png())
