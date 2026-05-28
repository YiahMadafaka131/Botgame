"""Torch Dataset over a build-dataset output dir (lazy torch import)."""

from __future__ import annotations

import json
import os

import numpy as np

from ..dataset.schema import Action, ActionType
from .encoding import action_to_target


def read_labels(dataset_dir: str) -> list[dict]:
    """Load labels.jsonl as a list of records."""
    path = os.path.join(dataset_dir, "labels.jsonl")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _record_to_action(rec: dict) -> Action:
    return Action(
        type=ActionType(rec["type"]),
        t=rec.get("t", 0.0),
        x=rec.get("x"),
        y=rec.get("y"),
        x2=rec.get("x2"),
        y2=rec.get("y2"),
    )


def build_torch_dataset(dataset_dir: str, screen_size: tuple[int, int]):
    """Return a torch Dataset yielding (frame_tensor, type_index, coords_tensor)."""
    import torch
    from torch.utils.data import Dataset

    width, height = screen_size
    records = read_labels(dataset_dir)

    class FrameActionDataset(Dataset):
        def __init__(self) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, i: int):
            rec = self.records[i]
            arr = np.load(os.path.join(dataset_dir, rec["frame"]))
            frame = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0
            idx, coords = action_to_target(_record_to_action(rec), width, height)
            return frame, idx, torch.tensor(coords, dtype=torch.float32)

    return FrameActionDataset()
