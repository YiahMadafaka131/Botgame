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


# --- confidence gating (pure, no torch) -------------------------------------

def test_select_action_index_confident_action():
    from botgame.model.encoding import select_action_index

    assert select_action_index([0.1, 0.8, 0.1], threshold=0.5) == 1


def test_select_action_index_uncertain_action_falls_back_to_noop():
    from botgame.model.encoding import select_action_index

    assert select_action_index([0.3, 0.4, 0.3], threshold=0.5) == 0


def test_select_action_index_noop_never_gated():
    from botgame.model.encoding import select_action_index

    assert select_action_index([0.4, 0.3, 0.3], threshold=0.9) == 0


def test_select_action_index_zero_threshold_is_argmax():
    from botgame.model.encoding import select_action_index

    assert select_action_index([0.34, 0.33, 0.33], threshold=0.0) == 0
    assert select_action_index([0.2, 0.39, 0.41], threshold=0.0) == 2


# --- class weighting (pure, no torch) ----------------------------------------

def test_class_weights_inverse_frequency():
    from botgame.model.train import class_weights

    w = class_weights([90, 9, 1])  # heavy noop majority
    assert w[2] > w[1] > w[0]  # rare classes weigh more
    assert sum(w) / 3 == pytest.approx(1.0)


def test_class_weights_handles_absent_class():
    from botgame.model.train import class_weights

    w = class_weights([10, 10, 0])
    assert w[0] == w[1] == 1.0
    assert w[2] == 1.0  # absent class: neutral weight, never used anyway


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

    # Geometry (input size, stack) comes from the checkpoint, not the caller.
    policy = Policy(ckpt, device="cpu")
    assert policy.input_size == (160, 90)
    assert policy.stack == 4
    frame = np.random.default_rng(1).integers(0, 256, size=(100, 200, 3), dtype=np.uint8)
    action = policy.predict(frame)
    assert action.type in (ActionType.NOOP, ActionType.TAP, ActionType.SWIPE)
    assert 0.0 <= policy.last_confidence <= 1.0
    if action.type is not ActionType.NOOP:
        assert 0 <= action.x <= 200
        assert 0 <= action.y <= 100


def test_frame_stacking_shapes(tmp_path):
    from botgame.model.dataset import build_torch_dataset

    ds_dir = str(tmp_path / "ds")
    _make_dataset(ds_dir)

    ds = build_torch_dataset(ds_dir, screen_size=(200, 100), stack=4)
    frame0, _, _ = ds[0]  # padded by repeating the first frame
    frame5, _, _ = ds[5]
    assert frame0.shape == (12, 90, 160)
    assert frame5.shape == (12, 90, 160)
    # Sample 0's stack is all copies of frame 0; channels must be identical.
    assert torch.equal(frame0[0:3], frame0[9:12])

    single = build_torch_dataset(ds_dir, screen_size=(200, 100), stack=1)
    frame, _, _ = single[0]
    assert frame.shape == (3, 90, 160)


def test_policy_bot_monitor_and_stop_on_anomaly(tmp_path):
    from botgame.bots.policy import PolicyBot, PolicyBotConfig
    from botgame.humanize import Humanizer
    from botgame.model.train import train
    from botgame.telemetry import TelemetryLogger

    ds_dir = str(tmp_path / "ds")
    _make_dataset(ds_dir)
    ckpt = str(tmp_path / "policy.pt")
    train(ds_dir, screen_size=(200, 100), epochs=1, batch_size=4, out_path=ckpt,
          device="cpu", val_split=0.0)

    class FakeCapture:
        rng = np.random.default_rng(3)

        def grab(self):
            return self.rng.integers(0, 256, size=(100, 200, 3), dtype=np.uint8)

    class FakeTouch:
        def tap(self, x, y): pass
        def swipe(self, *a, **k): pass
        def swipe_path(self, *a, **k): pass

    class StubMonitor:
        def __init__(self):
            self.calls = 0

        def check(self):
            self.calls += 1
            return ["anomaly"] if self.calls == 1 else []

    monitor = StubMonitor()
    config = PolicyBotConfig(
        interval_s=0.0, max_steps=10, check_every=2, stop_on_anomaly=True
    )
    with TelemetryLogger(str(tmp_path / "tel.jsonl")) as tel:
        bot = PolicyBot.__new__(PolicyBot)
        bot.capture = FakeCapture()
        bot.touch = FakeTouch()
        bot.humanizer = Humanizer(level=0.0)
        bot.telemetry = tel
        bot.config = config
        bot.monitor = monitor
        bot.anomalies = []
        from botgame.model.infer import Policy

        bot.policy = Policy(ckpt, device="cpu")
        steps = bot.run()

    assert steps == 2  # stopped at the first health check
    assert bot.anomalies == ["anomaly"]
    assert monitor.calls == 1


def test_policy_buffer_reset(tmp_path):
    from botgame.model.infer import Policy
    from botgame.model.train import train

    ds_dir = str(tmp_path / "ds")
    _make_dataset(ds_dir)
    ckpt = str(tmp_path / "policy.pt")
    train(ds_dir, screen_size=(200, 100), epochs=1, batch_size=4, out_path=ckpt,
          device="cpu", val_split=0.0)

    policy = Policy(ckpt, device="cpu", act_threshold=0.99)
    rng = np.random.default_rng(2)
    for _ in range(5):
        policy.predict(rng.integers(0, 256, size=(100, 200, 3), dtype=np.uint8))
    assert len(policy._frames) == 4
    policy.reset()
    assert len(policy._frames) == 0
