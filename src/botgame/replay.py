"""Deterministic replay of recorded actions ("game replicator").

Block 4: take the actions recovered from a gameplay video (getevent log,
build-dataset labels, or the show-touches overlay) and re-inject them on a
device with the original timing. This is the regression-testing mode: the same
run can be repeated loop after loop while a HealthMonitor watches for crashes,
ANRs and frozen screens.

The replay is exact at humanization level 0; raising the level adds the same
jitter/curvature/timing noise as the live bots, which is useful to check that
a bug reproduces under slightly-varying input too.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable

from .adb.input import TouchInput
from .dataset.schema import Action, ActionType
from .humanize import Humanizer
from .telemetry import TelemetryLogger


def load_actions_jsonl(path: str) -> list[Action]:
    """Load Actions from a JSONL file (e.g. a build-dataset labels.jsonl).

    NOOP records (frames where nothing happened) are skipped; the result is a
    time-ordered list of taps and swipes.
    """
    actions: list[Action] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            atype = ActionType(rec["type"])
            if atype is ActionType.NOOP:
                continue
            actions.append(
                Action(
                    type=atype,
                    t=float(rec["t"]),
                    x=rec.get("x"),
                    y=rec.get("y"),
                    x2=rec.get("x2"),
                    y2=rec.get("y2"),
                )
            )
    actions.sort(key=lambda a: a.t)
    return actions


def rescale_actions(
    actions: list[Action],
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> list[Action]:
    """Map action coordinates from one screen-pixel space to another.

    Lets a recording made on one device replay on a device with a different
    resolution. Aspect-ratio differences are stretched, not letterboxed.
    """
    sw, sh = src_size
    dw, dh = dst_size
    if sw <= 0 or sh <= 0:
        raise ValueError(f"src_size must be positive, got {src_size}")

    def fx(v: int | None) -> int | None:
        return None if v is None else int(round(v * dw / sw))

    def fy(v: int | None) -> int | None:
        return None if v is None else int(round(v * dh / sh))

    return [
        Action(type=a.type, t=a.t, x=fx(a.x), y=fy(a.y), x2=fx(a.x2), y2=fy(a.y2))
        for a in actions
    ]


@dataclass
class ReplayConfig:
    speed: float = 1.0  # 2.0 = twice as fast as the recording
    loops: int = 1  # repeat the whole sequence N times (stress testing)
    swipe_duration_ms: int = 250  # getevent does not preserve gesture duration
    check_every: int = 5  # health-check every N actions (0 = never)
    stop_on_anomaly: bool = False  # abort the replay when the monitor fires


@dataclass
class ReplayResult:
    actions_played: int = 0
    loops_completed: int = 0
    anomalies: list = field(default_factory=list)  # list[Anomaly]
    aborted: bool = False


class ReplayBot:
    """Re-inject a recorded action sequence, preserving relative timing.

    `monitor` is an optional object with a `check() -> list` method (see
    `botgame.monitor.HealthMonitor`); anything it returns is collected into
    the result and, with `stop_on_anomaly`, aborts the run.

    `sleep` and `clock` are injectable for tests.
    """

    def __init__(
        self,
        touch: TouchInput,
        actions: list[Action],
        humanizer: Humanizer,
        telemetry: TelemetryLogger,
        monitor: object | None = None,
        config: ReplayConfig | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not actions:
            raise ValueError("no actions to replay")
        self.touch = touch
        self.actions = sorted(actions, key=lambda a: a.t)
        self.humanizer = humanizer
        self.telemetry = telemetry
        self.monitor = monitor
        self.config = config or ReplayConfig()
        if self.config.speed <= 0:
            raise ValueError(f"speed must be positive, got {self.config.speed}")
        self._sleep = sleep
        self._clock = clock

    def _inject(self, action: Action) -> dict:
        """Perform one action through the humanizer; return telemetry fields."""
        if action.type is ActionType.TAP:
            jx, jy = self.humanizer.jitter_point(action.x, action.y)
            self.touch.tap(jx, jy)
            return {"action": "tap", "target": [action.x, action.y], "actual": [jx, jy]}
        if action.type is ActionType.SWIPE:
            if self.humanizer.level > 0:
                path = self.humanizer.bezier_path(
                    (action.x, action.y), (action.x2, action.y2)
                )
                self.touch.swipe_path(path, duration_ms=self.config.swipe_duration_ms)
                return {
                    "action": "swipe",
                    "start": [action.x, action.y],
                    "end": [action.x2, action.y2],
                    "points": len(path),
                }
            self.touch.swipe(
                action.x, action.y, action.x2, action.y2, self.config.swipe_duration_ms
            )
            return {
                "action": "swipe",
                "start": [action.x, action.y],
                "end": [action.x2, action.y2],
            }
        return {"action": "noop"}

    def run(self) -> ReplayResult:
        result = ReplayResult()
        cfg = self.config
        t0 = self.actions[0].t
        try:
            for loop in range(cfg.loops):
                loop_start = self._clock()
                for action in self.actions:
                    due = loop_start + (action.t - t0) / cfg.speed
                    due += self.humanizer.reaction_delay()
                    wait = due - self._clock()
                    if wait > 0:
                        self._sleep(wait)

                    fields = self._inject(action)
                    fields.update(
                        loop=loop, t_rec=round(action.t, 4), level=self.humanizer.level
                    )
                    self.telemetry.log(fields.pop("action"), **fields)
                    result.actions_played += 1

                    if (
                        self.monitor is not None
                        and cfg.check_every > 0
                        and result.actions_played % cfg.check_every == 0
                    ):
                        found = self.monitor.check()
                        result.anomalies.extend(found)
                        if found and cfg.stop_on_anomaly:
                            result.aborted = True
                            return result
                result.loops_completed = loop + 1
        except KeyboardInterrupt:
            result.aborted = True
        # Final sweep so a crash caused by the very last action is not missed.
        if self.monitor is not None and not result.aborted:
            result.anomalies.extend(self.monitor.check())
        return result
