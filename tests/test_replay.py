"""Tests for the replay block: action loading, rescaling, timing and the
health monitor. No device needed — ADB-facing pieces are faked."""

import json

import numpy as np
import pytest

from botgame.dataset.schema import Action, ActionType
from botgame.humanize import Humanizer
from botgame.monitor import Anomaly, HealthMonitor, foreground_package, frame_delta
from botgame.replay import ReplayBot, ReplayConfig, load_actions_jsonl, rescale_actions
from botgame.telemetry import TelemetryLogger


# --- fakes -----------------------------------------------------------------

class FakeTouch:
    def __init__(self):
        self.calls = []

    def tap(self, x, y):
        self.calls.append(("tap", x, y))

    def swipe(self, x1, y1, x2, y2, duration_ms=200):
        self.calls.append(("swipe", x1, y1, x2, y2, duration_ms))

    def swipe_path(self, points, duration_ms=300):
        self.calls.append(("swipe_path", points[0], points[-1]))


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


class StubMonitor:
    """Returns scripted anomalies on the nth call to check()."""

    def __init__(self, hits=None):
        self.calls = 0
        self.hits = hits or {}

    def check(self):
        self.calls += 1
        return self.hits.get(self.calls, [])


def make_bot(tmp_path, actions, monitor=None, level=0.0, **cfg):
    touch = FakeTouch()
    clock = FakeClock()
    tel = TelemetryLogger(str(tmp_path / "tel.jsonl"))
    bot = ReplayBot(
        touch,
        actions,
        Humanizer(level=level, seed=1),
        tel,
        monitor=monitor,
        config=ReplayConfig(**cfg),
        sleep=clock.sleep,
        clock=clock,
    )
    return bot, touch, clock


# --- loading & rescaling ----------------------------------------------------

def test_load_actions_jsonl_skips_noops_and_sorts(tmp_path):
    path = tmp_path / "labels.jsonl"
    records = [
        {"frame": "frames/0.npy", "type": "noop", "t": 0.0},
        {"frame": "frames/2.npy", "type": "swipe", "t": 2.0, "x": 1, "y": 2, "x2": 3, "y2": 4},
        {"frame": "frames/1.npy", "type": "tap", "t": 1.0, "x": 10, "y": 20},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    actions = load_actions_jsonl(str(path))

    assert [a.type for a in actions] == [ActionType.TAP, ActionType.SWIPE]
    assert [a.t for a in actions] == [1.0, 2.0]
    assert (actions[1].x2, actions[1].y2) == (3, 4)


def test_rescale_actions():
    acts = [Action.tap(0.0, 540, 1200), Action.swipe(1.0, 0, 0, 1080, 2400)]
    out = rescale_actions(acts, (1080, 2400), (720, 1600))
    assert (out[0].x, out[0].y) == (360, 800)
    assert (out[1].x2, out[1].y2) == (720, 1600)
    assert out[1].x2 is not None and out[0].x2 is None


def test_rescale_rejects_bad_src():
    with pytest.raises(ValueError):
        rescale_actions([Action.tap(0, 1, 1)], (0, 100), (100, 100))


# --- replay timing & injection ----------------------------------------------

def test_replay_preserves_order_and_speed(tmp_path):
    actions = [
        Action.tap(0.0, 100, 200),
        Action.tap(1.0, 110, 210),
        Action.swipe(2.0, 10, 10, 500, 500),
    ]
    bot, touch, clock = make_bot(tmp_path, actions, speed=2.0)
    result = bot.run()

    assert result.actions_played == 3
    assert result.loops_completed == 1
    assert [c[0] for c in touch.calls] == ["tap", "tap", "swipe"]
    # 2 s of recording at 2x speed -> last action due at t=1.0
    assert clock.t == pytest.approx(1.0)
    # Exact replica at level 0: no jitter on coordinates.
    assert touch.calls[0][1:] == (100, 200)


def test_replay_loops_and_monitor_cadence(tmp_path):
    actions = [Action.tap(0.0, 1, 1), Action.tap(0.1, 2, 2), Action.tap(0.2, 3, 3)]
    monitor = StubMonitor()
    bot, touch, _ = make_bot(tmp_path, actions, monitor=monitor, loops=2, check_every=2)
    result = bot.run()

    assert result.actions_played == 6
    assert result.loops_completed == 2
    # Checks after actions 2, 4, 6 plus the final sweep.
    assert monitor.calls == 4
    assert result.anomalies == []


def test_replay_stop_on_anomaly(tmp_path):
    crash = Anomaly(kind="crash", detail="FATAL EXCEPTION", ts=0.0)
    monitor = StubMonitor(hits={1: [crash]})
    actions = [Action.tap(0.0, 1, 1), Action.tap(0.1, 2, 2), Action.tap(0.2, 3, 3)]
    bot, touch, _ = make_bot(
        tmp_path, actions, monitor=monitor, check_every=2, stop_on_anomaly=True
    )
    result = bot.run()

    assert result.aborted
    assert result.actions_played == 2
    assert result.anomalies == [crash]


def test_replay_humanized_swipe_uses_path(tmp_path):
    actions = [Action.swipe(0.0, 0, 0, 100, 100)]
    bot, touch, _ = make_bot(tmp_path, actions, level=1.0)
    bot.run()
    kind, start, end = touch.calls[0]
    assert kind == "swipe_path"
    assert start == (0, 0) and end == (100, 100)  # endpoints pinned


def test_replay_rejects_empty_and_bad_speed(tmp_path):
    with pytest.raises(ValueError):
        make_bot(tmp_path, [])
    with pytest.raises(ValueError):
        make_bot(tmp_path, [Action.tap(0, 1, 1)], speed=0.0)


# --- monitor ------------------------------------------------------------------

def test_foreground_package_parsing():
    text = (
        "  mResumedActivity: ActivityRecord{abc123 u0 "
        "com.example.game/.MainActivity t42}\n"
    )
    assert foreground_package(text) == "com.example.game"
    assert foreground_package("no activities here") is None


def test_frame_delta():
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b = a.copy()
    assert frame_delta(a, b) == 0.0
    b[0, 0] = 255
    assert frame_delta(a, b) > 0.0
    assert frame_delta(a, np.zeros((2, 2, 3), dtype=np.uint8)) == 255.0


class FakeHealthMonitor(HealthMonitor):
    def __init__(self, report_dir, package=None, **kw):
        self.crash_lines = []
        self.fg_text = ""
        self.frame = np.zeros((4, 4, 3), dtype=np.uint8)
        super().__init__(device=object(), package=package, report_dir=report_dir, **kw)

    def _crash_log(self):
        return list(self.crash_lines)

    def _dumpsys_activities(self):
        return self.fg_text

    def _grab(self):
        return self.frame.copy()


def test_monitor_detects_new_crash_only_once(tmp_path):
    mon = FakeHealthMonitor(str(tmp_path / "rep"))
    mon.crash_lines = ["06-10 FATAL EXCEPTION: main", "  at com.example.Game.boom"]
    found = mon.check()
    assert [a.kind for a in found] == ["crash"]
    assert "FATAL EXCEPTION" in found[0].detail
    # Already-seen lines do not re-trigger.
    assert mon.check() == []


def test_monitor_detects_lost_focus(tmp_path):
    mon = FakeHealthMonitor(str(tmp_path / "rep"), package="com.example.game")
    mon.fg_text = "mResumedActivity: ActivityRecord{x u0 com.example.game/.Main t1}"
    assert mon.check() == []
    mon.fg_text = "mResumedActivity: ActivityRecord{x u0 com.android.launcher/.Home t2}"
    found = mon.check()
    assert [a.kind for a in found] == ["lost_focus"]
    assert "com.android.launcher" in found[0].detail


def test_monitor_detects_frozen_screen_and_persists(tmp_path):
    report = tmp_path / "rep"
    mon = FakeHealthMonitor(str(report), freeze_checks=2)
    assert mon.check() == []  # first frame only sets the baseline
    assert mon.check() == []  # static_count = 1
    found = mon.check()  # static_count = 2 -> freeze
    assert [a.kind for a in found] == ["frozen_screen"]
    assert mon.check() == []  # reported once per episode

    # Evidence persisted: screenshot + report.jsonl line.
    assert found[0].screenshot is not None
    assert (report / "report.jsonl").exists()
    rec = json.loads((report / "report.jsonl").read_text().splitlines()[0])
    assert rec["kind"] == "frozen_screen"

    # Screen changes -> episode resets and can fire again later.
    mon.frame = np.full((4, 4, 3), 200, dtype=np.uint8)
    assert mon.check() == []
    mon.check()
    found = mon.check()
    assert [a.kind for a in found] == ["frozen_screen"]
