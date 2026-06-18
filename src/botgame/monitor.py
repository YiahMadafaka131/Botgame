"""Session health monitoring: turn a replay into a bug hunt.

While a bot replays (or plays) a game, the HealthMonitor periodically checks
three signals and records an Anomaly for each finding:

  - crash/ANR:    new entries in the logcat crash buffer (`logcat -d -b crash`)
  - lost focus:   the game is no longer the resumed activity (crashed to the
                  launcher, an ANR dialog stole focus, an ad opened a browser…)
  - frozen screen: N consecutive captures that are pixel-wise (near) identical
                  while the bot is actively injecting input

Each anomaly saves a screenshot into the report directory and appends a JSON
record to `report.jsonl`, so a failing run leaves reviewable evidence.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass

import numpy as np

from .adb import AdbDevice, ScreenCapture
from .adb.device import AdbError

_RESUMED = re.compile(
    r"(?:mResumedActivity|mFocusedApp|topResumedActivity)[^{]*\{[^}]*?\s"
    r"(?P<pkg>[A-Za-z][\w.]*)/(?P<activity>[\w.$]+)"
)
_CRASH_MARKERS = ("FATAL EXCEPTION", "ANR in", "Fatal signal", "DEBUG : *** ***")


@dataclass
class Anomaly:
    kind: str  # "crash" | "lost_focus" | "frozen_screen"
    detail: str
    ts: float
    screenshot: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def foreground_package(dumpsys_text: str) -> str | None:
    """Extract the resumed/focused package from `dumpsys activity activities`."""
    m = _RESUMED.search(dumpsys_text)
    return m.group("pkg") if m else None


def frame_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute pixel difference between two frames (0 = identical)."""
    if a.shape != b.shape:
        return 255.0  # resolution change (rotation?) is definitely not frozen
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


class HealthMonitor:
    """Checks device health between bot actions; see module docstring.

    `package` is the app under test; leave it None to skip the focus check.
    `freeze_checks` consecutive near-identical frames (delta below
    `freeze_threshold`) raise a frozen-screen anomaly once per freeze episode.
    """

    def __init__(
        self,
        device: AdbDevice,
        package: str | None = None,
        report_dir: str = "bugreport",
        freeze_threshold: float = 0.5,
        freeze_checks: int = 3,
    ):
        self.device = device
        self.capture = ScreenCapture(device)
        self.package = package
        self.report_dir = report_dir
        self.freeze_threshold = freeze_threshold
        self.freeze_checks = freeze_checks
        self._last_frame: np.ndarray | None = None
        self._static_count = 0
        self._frozen_reported = False
        self._crash_lines_seen = len(self._crash_log())

    # --- device probes (overridable in tests) -----------------------------
    def _crash_log(self) -> list[str]:
        try:
            return self.device.shell("logcat", "-d", "-b", "crash").splitlines()
        except AdbError:
            return []  # some devices lack the crash buffer; skip the check

    def _dumpsys_activities(self) -> str:
        return self.device.shell("dumpsys", "activity", "activities")

    def _grab(self) -> np.ndarray | None:
        try:
            return self.capture.grab()
        except AdbError:
            return None

    # --- checks ------------------------------------------------------------
    def _check_crash(self) -> Anomaly | None:
        lines = self._crash_log()
        new = lines[self._crash_lines_seen :]
        self._crash_lines_seen = len(lines)
        hits = [ln for ln in new if any(marker in ln for marker in _CRASH_MARKERS)]
        if not hits:
            return None
        return Anomaly(kind="crash", detail="\n".join(new[-40:]), ts=time.time())

    def _check_focus(self) -> Anomaly | None:
        if not self.package:
            return None
        fg = foreground_package(self._dumpsys_activities())
        if fg is None or fg == self.package:
            return None
        return Anomaly(
            kind="lost_focus",
            detail=f"expected {self.package} in foreground, found {fg}",
            ts=time.time(),
        )

    def _check_frozen(self, frame: np.ndarray | None) -> Anomaly | None:
        if frame is None:
            return None
        prev, self._last_frame = self._last_frame, frame
        if prev is None:
            return None
        if frame_delta(prev, frame) < self.freeze_threshold:
            self._static_count += 1
        else:
            self._static_count = 0
            self._frozen_reported = False
        if self._static_count >= self.freeze_checks and not self._frozen_reported:
            self._frozen_reported = True  # one report per freeze episode
            return Anomaly(
                kind="frozen_screen",
                detail=f"screen unchanged across {self._static_count + 1} checks",
                ts=time.time(),
            )
        return None

    # --- public API ----------------------------------------------------------
    def check(self) -> list[Anomaly]:
        """Run all checks once; persist and return any anomalies found."""
        frame = self._grab()
        anomalies = [
            a
            for a in (
                self._check_crash(),
                self._check_focus(),
                self._check_frozen(frame),
            )
            if a is not None
        ]
        for anomaly in anomalies:
            self._persist(anomaly, frame)
        return anomalies

    def _persist(self, anomaly: Anomaly, frame: np.ndarray | None) -> None:
        os.makedirs(self.report_dir, exist_ok=True)
        if frame is not None:
            from PIL import Image

            name = f"{anomaly.kind}_{int(anomaly.ts * 1000)}.png"
            path = os.path.join(self.report_dir, name)
            Image.fromarray(frame).save(path)
            anomaly.screenshot = path
        with open(
            os.path.join(self.report_dir, "report.jsonl"), "a", encoding="utf-8"
        ) as fh:
            fh.write(json.dumps(anomaly.to_dict()) + "\n")
