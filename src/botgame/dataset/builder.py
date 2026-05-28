"""Align frames with actions and write an imitation-learning dataset.

Given frame timestamps and a list of timestamped Actions, each frame is
labelled with the action that occurred during its time window, or NOOP. The
result is a directory of downscaled frame .npy arrays plus a labels.jsonl.
"""

from __future__ import annotations

import bisect
import json
import os

import numpy as np

from .schema import Action, ActionType


def align_actions(
    frame_times: list[float],
    actions: list[Action],
    *,
    fps: float,
) -> list[Action]:
    """Label each frame time with the action falling in [t, t + 1/fps), else NOOP.

    If several actions fall in one window the earliest is kept. Frame times must
    be sorted ascending.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    window = 1.0 / fps
    act_times = [a.t for a in actions]
    labels: list[Action] = []
    for ft in frame_times:
        lo = bisect.bisect_left(act_times, ft)
        hi = bisect.bisect_left(act_times, ft + window)
        if hi > lo:
            labels.append(actions[lo])
        else:
            labels.append(Action.noop(ft))
    return labels


def _downscale(frame_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize to (W, H) using numpy only (no OpenCV)."""
    w, h = size
    sh, sw = frame_rgb.shape[:2]
    ys = (np.linspace(0, sh - 1, h)).astype(np.intp)
    xs = (np.linspace(0, sw - 1, w)).astype(np.intp)
    return frame_rgb[ys][:, xs]


def build_dataset(
    frames: "object",
    actions: list[Action],
    out_dir: str,
    *,
    fps: float,
    resize: tuple[int, int] | None = (160, 90),
) -> int:
    """Write a dataset from a (timestamp, frame_rgb) iterable and actions.

    Returns the number of samples written. Each sample i produces
    `frames/{i:06d}.npy` and a line in `labels.jsonl`.
    """
    frame_dir = os.path.join(out_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    times: list[float] = []
    paths: list[str] = []
    for i, (t, frame) in enumerate(frames):
        arr = _downscale(frame, resize) if resize else frame
        p = os.path.join(frame_dir, f"{i:06d}.npy")
        np.save(p, arr)
        times.append(t)
        paths.append(p)

    labels = align_actions(times, actions, fps=fps)

    with open(os.path.join(out_dir, "labels.jsonl"), "w", encoding="utf-8") as fh:
        for p, label in zip(paths, labels):
            rec = {"frame": os.path.relpath(p, out_dir), **label.to_dict()}
            fh.write(json.dumps(rec) + "\n")

    n_actions = sum(1 for label in labels if label.type is not ActionType.NOOP)
    print(f"Wrote {len(paths)} samples ({n_actions} action / {len(paths) - n_actions} noop)")
    return len(paths)
