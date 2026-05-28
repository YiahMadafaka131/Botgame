import json
import os

import numpy as np
import pytest

from botgame.dataset.schema import Action, ActionType
from botgame.model.encoding import action_to_target, decode_prediction


def test_tap_target_roundtrip():
    idx, coords = action_to_target(Action.tap(0.0, 100, 50), width=200, height=100)
    assert idx == 1  # TAP
    assert coords == [0.5, 0.5, 0.5, 0.5]
    back = decode_prediction(idx, coords, width=200, height=100)
    assert back.type is ActionType.TAP
    assert (back.x, back.y) == (100, 50)


def test_swipe_target_roundtrip():
    idx, coords = action_to_target(Action.swipe(0.0, 0, 0, 200, 100), 200, 100)
    assert idx == 2  # SWIPE
    assert coords == [0.0, 0.0, 1.0, 1.0]
    back = decode_prediction(idx, coords, 200, 100)
    assert back.type is ActionType.SWIPE
    assert (back.x, back.y, back.x2, back.y2) == (0, 0, 200, 100)


def test_noop_has_zero_coords():
    idx, coords = action_to_target(Action.noop(0.0), 200, 100)
    assert idx == 0
    assert coords == [0.0, 0.0, 0.0, 0.0]


def test_coords_are_clamped():
    idx, coords = action_to_target(Action.tap(0.0, 999, -5), 200, 100)
    assert coords[0] == 1.0  # clamped high
    assert coords[1] == 0.0  # clamped low


def test_target_rejects_bad_size():
    with pytest.raises(ValueError):
        action_to_target(Action.tap(0.0, 1, 1), 0, 100)


# --- torch end-to-end smoke test ------------------------------------------
torch = pytest.importorskip("torch")


def _make_dataset(root: str, n: int = 6) -> None:
    frame_dir = os.path.join(root, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    with open(os.path.join(root, "labels.jsonl"), "w", encoding="utf-8") as fh:
        for i in range(n):
            arr = rng.integers(0, 256, size=(90, 160, 3), dtype=np.uint8)
            rel = os.path.join("frames", f"{i:06d}.npy")
            np.save(os.path.join(root, rel), arr)
            if i % 3 == 0:
                rec = {"frame": rel, "type": "noop", "t": float(i)}
            elif i % 3 == 1:
                rec = {"frame": rel, "type": "tap", "t": float(i), "x": 100, "y": 50}
            else:
                rec = {"frame": rel, "type": "swipe", "t": float(i),
                       "x": 10, "y": 10, "x2": 180, "y2": 90}
            fh.write(json.dumps(rec) + "\n")


def test_train_and_predict_smoke(tmp_path):
    from botgame.model.infer import Policy
    from botgame.model.train import train

    ds_dir = str(tmp_path / "ds")
    _make_dataset(ds_dir)
    ckpt = str(tmp_path / "policy.pt")

    train(ds_dir, screen_size=(200, 100), epochs=1, batch_size=4, out_path=ckpt, device="cpu")
    assert os.path.exists(ckpt)

    policy = Policy(ckpt, input_size=(160, 90), device="cpu")
    frame = np.random.default_rng(1).integers(0, 256, size=(100, 200, 3), dtype=np.uint8)
    action = policy.predict(frame)
    assert action.type in (ActionType.NOOP, ActionType.TAP, ActionType.SWIPE)
    if action.type is not ActionType.NOOP:
        assert 0 <= action.x <= 200
        assert 0 <= action.y <= 100
