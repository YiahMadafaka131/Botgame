"""Load a trained policy and predict Actions from frames (lazy torch import)."""

from __future__ import annotations

from collections import deque

import numpy as np

from ..dataset.schema import Action
from .encoding import decode_prediction, select_action_index


def _resize_nn(frame_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    sh, sw = frame_rgb.shape[:2]
    ys = np.linspace(0, sh - 1, h).astype(np.intp)
    xs = np.linspace(0, sw - 1, w).astype(np.intp)
    return frame_rgb[ys][:, xs]


class Policy:
    """Wraps a trained net; `predict(frame_rgb)` returns an Action in screen px.

    Model geometry (input size, frame-stack depth) is read from the checkpoint,
    so it always matches training; `input_size` is only a fallback for old
    checkpoints that did not store it. The policy keeps a rolling buffer of the
    last `stack` frames internally — call `reset()` when the game restarts.

    `act_threshold` gates actions on softmax confidence: a non-NOOP prediction
    below the threshold becomes a NOOP, so the bot waits instead of guessing.
    The confidence of the latest prediction is exposed as `last_confidence`.
    """

    def __init__(
        self,
        checkpoint: str,
        input_size: tuple[int, int] = (160, 90),
        device: str | None = None,
        act_threshold: float = 0.0,
    ):
        import torch

        from .net import build_policy_net

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint, map_location=self.device)
        self.input_size = tuple(ckpt.get("input_size", input_size))
        self.stack = int(ckpt.get("stack", 1))
        self.act_threshold = act_threshold
        self.last_confidence = 0.0
        self._frames: deque[np.ndarray] = deque(maxlen=self.stack)
        self.net = build_policy_net(in_channels=int(ckpt.get("in_channels", 3)))
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.to(self.device).eval()

    def reset(self) -> None:
        """Clear the frame buffer (e.g. after the game restarts)."""
        self._frames.clear()

    def predict(self, frame_rgb: np.ndarray) -> Action:
        h, w = frame_rgb.shape[:2]
        small = _resize_nn(frame_rgb, self.input_size)
        self._frames.append(small)
        # Pad by repeating the oldest frame until the buffer fills up.
        stacked = list(self._frames)
        stacked = [stacked[0]] * (self.stack - len(stacked)) + stacked
        arr = np.concatenate(stacked, axis=2) if self.stack > 1 else stacked[0]

        tensor = (
            self.torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        ).to(self.device)
        with self.torch.no_grad():
            type_logits, coord_pred = self.net(tensor)
        probs = self.torch.softmax(type_logits, dim=1).squeeze(0).tolist()
        type_index = select_action_index(probs, self.act_threshold)
        self.last_confidence = probs[type_index]
        coords = coord_pred.squeeze(0).tolist()
        return decode_prediction(type_index, coords, w, h)
