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
    input_size: tuple[int, int] = (160, 90)


class PolicyBot:
    def __init__(
        self,
        device: AdbDevice,
        checkpoint: str,
        humanizer: Humanizer,
        telemetry: TelemetryLogger,
        config: PolicyBotConfig | None = None,
    ):
        self.device = device
        self.capture = ScreenCapture(device)
        self.touch = TouchInput(device)
        self.humanizer = humanizer
        self.telemetry = telemetry
        self.config = config or PolicyBotConfig()
        self.policy = Policy(checkpoint, input_size=self.config.input_size)

    def step(self) -> None:
        frame = self.capture.grab()
        action = self.policy.predict(frame)

        if action.type is ActionType.NOOP:
            self.telemetry.log("noop", level=self.humanizer.level)
            return

        delay = self.humanizer.reaction_delay()
        if delay > 0:
            time.sleep(delay)

        if action.type is ActionType.TAP:
            jx, jy = self.humanizer.jitter_point(action.x, action.y)
            self.touch.tap(jx, jy)
            self.telemetry.log(
                "tap", target=[action.x, action.y], actual=[jx, jy],
                reaction_s=round(delay, 4), level=self.humanizer.level,
            )
        else:  # SWIPE
            path = self.humanizer.bezier_path((action.x, action.y), (action.x2, action.y2))
            self.touch.swipe_path(path)
            self.telemetry.log(
                "swipe", start=[action.x, action.y], end=[action.x2, action.y2],
                points=len(path), reaction_s=round(delay, 4), level=self.humanizer.level,
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
