"""A placeholder bot: taps random points inside a region at a steady cadence.

It carries no perception or learned policy yet — its job is to exercise the
full pipeline (capture -> decide -> humanize -> inject -> log) end to end so
you can point your detector at it. Swap `_choose_target` for a real
perception+policy module when the imitation-learning model is ready.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..adb import AdbDevice, ScreenCapture, TouchInput
from ..humanize import Humanizer
from ..telemetry import TelemetryLogger

Region = tuple[int, int, int, int]  # x, y, width, height


@dataclass
class DummyBotConfig:
    region: Region | None = None  # None -> full screen
    interval_s: float = 0.8  # nominal seconds between actions
    max_steps: int = 50  # 0 = run until interrupted
    use_swipes: bool = False  # tap (False) or curved swipe (True)
    capture_frames: bool = True  # grab a frame each step (proves capture works)
    seed: int | None = None


class DummyBot:
    def __init__(
        self,
        device: AdbDevice,
        humanizer: Humanizer,
        telemetry: TelemetryLogger,
        config: DummyBotConfig | None = None,
        capture=None,
        touch=None,
    ):
        self.device = device
        self.capture = capture if capture is not None else ScreenCapture(device)
        self.touch = touch if touch is not None else TouchInput(device)
        self.humanizer = humanizer
        self.telemetry = telemetry
        self.config = config or DummyBotConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._region = self._resolve_region()

    def _resolve_region(self) -> Region:
        if self.config.region is not None:
            return self.config.region
        w, h = self.device.screen_size()
        return (0, 0, w, h)

    def _choose_target(self) -> tuple[int, int]:
        x, y, w, h = self._region
        return int(self._rng.integers(x, x + w)), int(self._rng.integers(y, y + h))

    def step(self) -> None:
        if self.config.capture_frames:
            frame = self.capture.grab()  # (H, W, 3); a real policy reads this
            del frame

        tx, ty = self._choose_target()

        delay = self.humanizer.reaction_delay()
        if delay > 0:
            time.sleep(delay)

        if self.config.use_swipes:
            sx, sy = self._choose_target()
            path = self.humanizer.bezier_path((sx, sy), (tx, ty))
            self.touch.swipe_path(path)
            self.telemetry.log(
                "swipe",
                start=[sx, sy],
                end=[tx, ty],
                points=len(path),
                reaction_s=round(delay, 4),
                level=self.humanizer.level,
            )
        else:
            jx, jy = self.humanizer.jitter_point(tx, ty)
            self.touch.tap(jx, jy)
            self.telemetry.log(
                "tap",
                target=[tx, ty],
                actual=[jx, jy],
                reaction_s=round(delay, 4),
                level=self.humanizer.level,
            )

    def run(self) -> int:
        steps = 0
        try:
            while self.config.max_steps == 0 or steps < self.config.max_steps:
                self.step()
                steps += 1
                time.sleep(self.humanizer.action_interval(self.config.interval_s))
        except KeyboardInterrupt:
            pass
        return steps
