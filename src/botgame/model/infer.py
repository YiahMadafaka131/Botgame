"""Load a trained policy and predict Actions from frames (lazy torch import)."""

from __future__ import annotations

import numpy as np

from ..dataset.schema import Action
from .encoding import decode_prediction
from .net import build_policy_net


def _resize_nn(frame_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    sh, sw = frame_rgb.shape[:2]
    ys = np.linspace(0, sh - 1, h).astype(np.intp)
    xs = np.linspace(0, sw - 1, w).astype(np.intp)
    return frame_rgb[ys][:, xs]


class Policy:
    """Wraps a trained net; `predict(frame_rgb)` returns an Action in screen px.

    `input_size` (W, H) must match the dataset's --resize used for training.
    """

    def __init__(self, checkpoint: str, input_size: tuple[int, int] = (160, 90),
                 device: str | None = None):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        ckpt = torch.load(checkpoint, map_location=self.device)
        self.net = build_policy_net()
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.to(self.device).eval()

    def predict(self, frame_rgb: np.ndarray) -> Action:
        h, w = frame_rgb.shape[:2]
        small = _resize_nn(frame_rgb, self.input_size)
        tensor = (
            self.torch.from_numpy(small).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        ).to(self.device)
        with self.torch.no_grad():
            type_logits, coord_pred = self.net(tensor)
        type_index = int(type_logits.argmax(dim=1).item())
        coords = coord_pred.squeeze(0).tolist()
        return decode_prediction(type_index, coords, w, h)
