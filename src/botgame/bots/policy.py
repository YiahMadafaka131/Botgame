"""Bot driven by a trained policy network.

Same loop as the dummy bot, but the action comes from the model's prediction on
the live frame instead of a random target. Humanization and telemetry are
applied identically, so detector experiments stay comparable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..adb import AdbDevice, ScreenCapture, TouchInput
from ..dataset.schema import ActionType
from ..humanize import Humanizer
from ..model.infer import Policy
from ..telemetry import TelemetryLogger


@dataclass
class PolicyBotConfig:
    interval_s: float = 0.2
    max_steps: int = 0  # 0 = run until interrupted
    input_size: tuple[int, int] = (160, 90)  # fallback for old checkpoints
    act_threshold: float = 0.5  # min confidence to act (0 = always trust argmax)
    check_every: int = 10  # health-check every N steps (0 = never)
    stop_on_anomaly: bool = False


class PolicyBot:
    def __init__(
        self,
        device: AdbDevice,
        checkpoint: str,
        humanizer: Humanizer,
        telemetry: TelemetryLogger,
        config: PolicyBotConfig | None = None,
        capture=None,
        touch=None,
        monitor: object | None = None,
    ):
        self.device = device
        self.capture = capture if capture is not None else ScreenCapture(device)
        self.touch = touch if touch is not None else TouchInput(device)
        self.humanizer = humanizer
        self.telemetry = telemetry
        self.config = config or PolicyBotConfig()
        self.monitor = monitor
        self.anomalies: list = []
        self.policy = Policy(
            checkpoint,
            input_size=self.config.input_size,
            act_threshold=self.config.act_threshold,
        )

    def step(self) -> None:
        frame = self.capture.grab()
        action = self.policy.predict(frame)

        if action.type is ActionType.NOOP:
            self.telemetry.log(
                "noop",
                confidence=round(self.policy.last_confidence, 3),
                level=self.humanizer.level,
            )
            return

        delay = self.humanizer.reaction_delay()
        if delay > 0:
            time.sleep(delay)

        confidence = round(self.policy.last_confidence, 3)
        if action.type is ActionType.TAP:
            jx, jy = self.humanizer.jitter_point(action.x, action.y)
            self.touch.tap(jx, jy)
            self.telemetry.log(
                "tap", target=[action.x, action.y], actual=[jx, jy],
                confidence=confidence, reaction_s=round(delay, 4),
                level=self.humanizer.level,
            )
        else:  # SWIPE
            path = self.humanizer.bezier_path((action.x, action.y), (action.x2, action.y2))
            self.touch.swipe_path(path)
            self.telemetry.log(
                "swipe", start=[action.x, action.y], end=[action.x2, action.y2],
                points=len(path), confidence=confidence,
                reaction_s=round(delay, 4), level=self.humanizer.level,
            )

    def run(self) -> int:
        steps = 0
        try:
            while self.config.max_steps == 0 or steps < self.config.max_steps:
                self.step()
                steps += 1
                if (
                    self.monitor is not None
                    and self.config.check_every > 0
                    and steps % self.config.check_every == 0
                ):
                    found = self.monitor.check()
                    self.anomalies.extend(found)
                    if found and self.config.stop_on_anomaly:
                        break
                time.sleep(self.humanizer.action_interval(self.config.interval_s))
        except KeyboardInterrupt:
            pass
        return steps
