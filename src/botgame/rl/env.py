"""Environment abstraction for RL fine-tuning.

PPO rolls episodes by calling `env.reset()` and `env.step(action)` repeatedly.
We expose a minimal Gym-flavoured interface — `EnvStep` is the (frame, reward,
done, info) tuple — so plugging in your game-specific reward function does not
require implementing the full Gym API.

`BotEnv` wires capture + touch + reward together for live training. `RandomEnv`
is a CPU-only stub used by the unit tests so the trainer can be exercised
without a phone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from ..dataset.schema import Action, ActionType


@dataclass
class EnvStep:
    frame: np.ndarray
    reward: float
    done: bool
    info: dict


class Capture(Protocol):
    def grab(self) -> np.ndarray: ...


class Touch(Protocol):
    def tap(self, x: int, y: int) -> None: ...
    def swipe_path(self, points: list[tuple[int, int]], duration_ms: int = 300) -> None: ...


RewardFn = Callable[[np.ndarray, Action, np.ndarray], float]
"""Signature: (prev_frame, action, next_frame) -> reward."""


class BotEnv:
    """Live environment driven by the ADB pipeline + a user-provided reward fn."""

    def __init__(
        self,
        capture: Capture,
        touch: Touch,
        reward_fn: RewardFn,
        step_interval_s: float = 0.1,
        max_steps: int = 256,
        done_fn: Callable[[np.ndarray, Action, np.ndarray], bool] | None = None,
    ):
        self.capture = capture
        self.touch = touch
        self.reward_fn = reward_fn
        self.step_interval_s = step_interval_s
        self.max_steps = max_steps
        self.done_fn = done_fn or (lambda *_: False)
        self._last_frame: np.ndarray | None = None
        self._step_count = 0

    def reset(self) -> np.ndarray:
        self._step_count = 0
        self._last_frame = self.capture.grab()
        return self._last_frame

    def step(self, action: Action) -> EnvStep:
        if self._last_frame is None:
            raise RuntimeError("Call reset() before step()")
        if action.type is ActionType.TAP:
            self.touch.tap(int(action.x), int(action.y))
        elif action.type is ActionType.SWIPE:
            pts = [(int(action.x), int(action.y)), (int(action.x2), int(action.y2))]
            self.touch.swipe_path(pts)
        # NOOP: no touch dispatched.
        time.sleep(self.step_interval_s)
        next_frame = self.capture.grab()
        reward = float(self.reward_fn(self._last_frame, action, next_frame))
        self._step_count += 1
        done = self._step_count >= self.max_steps or self.done_fn(
            self._last_frame, action, next_frame
        )
        self._last_frame = next_frame
        return EnvStep(frame=next_frame, reward=reward, done=done, info={})


class RandomEnv:
    """Stub env emitting random frames; reward = sparse signal on TAP actions.

    Lets the PPO trainer be unit-tested without a device. The reward is shaped
    so that the policy genuinely has something to learn (TAP near the centre
    pays more than TAP elsewhere; NOOP pays a constant penalty).
    """

    def __init__(
        self,
        frame_shape: tuple[int, int, int] = (90, 160, 3),
        max_steps: int = 8,
        seed: int | None = None,
    ):
        self.frame_shape = frame_shape
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self._step_count = 0

    def reset(self) -> np.ndarray:
        self._step_count = 0
        return self._frame()

    def step(self, action: Action) -> EnvStep:
        self._step_count += 1
        if action.type is ActionType.TAP:
            cx, cy = self.frame_shape[1] / 2.0, self.frame_shape[0] / 2.0
            d = float(np.hypot(action.x - cx, action.y - cy))
            reward = 1.0 - d / (cx + cy)
        elif action.type is ActionType.SWIPE:
            reward = 0.1
        else:
            reward = -0.2
        return EnvStep(
            frame=self._frame(),
            reward=reward,
            done=self._step_count >= self.max_steps,
            info={},
        )

    def _frame(self) -> np.ndarray:
        return self.rng.integers(0, 256, size=self.frame_shape, dtype=np.uint8)
