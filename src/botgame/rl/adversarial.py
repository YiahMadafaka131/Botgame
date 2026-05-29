"""Closed-loop adversarial reward.

`detection_evasion_reward(scorer)` returns a `RewardFn` that maintains a
rolling buffer of the policy's recent actions and runs a row-based detector
scorer against it each step. The reward is `-scale * score`, so the policy
learns to behave less like a bot under your own detector.

The scorer receives action rows shaped like the telemetry JSONL (one dict per
action with `ts`, `action`, `target`, `actual`, `reaction_s`). Pass a
`Humanizer` to simulate the deployed pipeline — rows then carry jittered
`actual` coords and sampled `reaction_s`, so `perfect_aim_score` and
`reaction_time_score` produce a meaningful gradient instead of saturating.
Without a humanizer, `actual == target` and `reaction_s == 0`, which makes
those two detectors uninformative; stick to `coord_cluster_score` or a
custom scorer over policy-controllable signals.

Usage:

    from botgame.humanize import Humanizer
    from botgame.rl import compose_rewards, detection_evasion_reward
    from botgame.rl.rewards import pixel_diff_reward
    from botgame.redteam.detectors import default_composite_score

    reward = compose_rewards([
        (pixel_diff_reward(), 1.0),
        (detection_evasion_reward(default_composite_score, scale=0.5,
                                   humanizer=Humanizer(level=1.0, seed=0)),
         1.0),
    ])

Any `Callable[[list[dict]], float]` returning a score in `[0, 1]` works as a
scorer — including custom detectors you write yourself.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable

import numpy as np

from ..dataset.schema import Action, ActionType
from ..humanize import Humanizer
from .rewards import RewardFn

RowScorer = Callable[[list[dict]], float]


def detection_evasion_reward(
    scorer: RowScorer,
    *,
    scale: float = 1.0,
    buffer_size: int = 64,
    min_actions: int = 3,
    humanizer: Humanizer | None = None,
    clock: Callable[[], float] = time.time,
) -> RewardFn:
    """Reward that penalises policy behaviour matching a detector.

    Args:
        scorer: row-based detector; takes a list of telemetry-shaped dicts,
            returns a bot-likelihood score in [0, 1].
        scale: multiplier on the (negated) score before returning.
        buffer_size: how many recent action rows to feed the scorer.
        min_actions: skip scoring until the buffer has at least this many rows
            (avoids noise when the scorer needs samples to stabilise).
        humanizer: optional Humanizer used to simulate the deployed pipeline
            (jittered actual coords + sampled reaction_s) when building rows.
            With it, all built-in scorers become informative; without it,
            `actual == target` and `reaction_s == 0`.
        clock: timestamp source — overridable for tests so periodicity
            metrics are deterministic.
    """
    if buffer_size < 1:
        raise ValueError("buffer_size must be >= 1")
    buf: deque[dict] = deque(maxlen=buffer_size)

    def _r(_prev: np.ndarray, action: Action, _nxt: np.ndarray) -> float:
        row = _row_from_action(action, clock(), humanizer)
        if row is not None:
            buf.append(row)
        if len(buf) < min_actions:
            return 0.0
        return float(-scale * scorer(list(buf)))

    return _r


def _row_from_action(
    action: Action, ts: float, humanizer: Humanizer | None
) -> dict | None:
    if action.type is ActionType.NOOP:
        return {"ts": ts, "action": "noop"}
    if action.type is ActionType.TAP:
        if action.x is None or action.y is None:
            return None
        target = [int(action.x), int(action.y)]
        if humanizer is not None:
            jx, jy = humanizer.jitter_point(int(action.x), int(action.y))
            actual = [int(jx), int(jy)]
            reaction_s = float(humanizer.reaction_delay())
        else:
            actual = list(target)
            reaction_s = 0.0
        return {
            "ts": ts,
            "action": "tap",
            "target": target,
            "actual": actual,
            "reaction_s": reaction_s,
        }
    if action.type is ActionType.SWIPE:
        if None in (action.x, action.y, action.x2, action.y2):
            return None
        if humanizer is not None:
            reaction_s = float(humanizer.reaction_delay())
        else:
            reaction_s = 0.0
        return {
            "ts": ts,
            "action": "swipe",
            "start": [int(action.x), int(action.y)],
            "end": [int(action.x2), int(action.y2)],
            "reaction_s": reaction_s,
        }
    return None
