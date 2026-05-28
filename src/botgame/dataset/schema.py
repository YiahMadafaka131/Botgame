"""Data schema shared by the dataset pipeline.

An Action is one labelled decision aligned to a moment in the gameplay video.
The imitation-learning model will be trained to predict these from frames.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class ActionType(str, Enum):
    NOOP = "noop"
    TAP = "tap"
    SWIPE = "swipe"


@dataclass
class Action:
    type: ActionType
    t: float  # seconds, relative to the start of the recording
    x: int | None = None
    y: int | None = None
    x2: int | None = None
    y2: int | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        return d

    @staticmethod
    def noop(t: float) -> "Action":
        return Action(type=ActionType.NOOP, t=t)

    @staticmethod
    def tap(t: float, x: int, y: int) -> "Action":
        return Action(type=ActionType.TAP, t=t, x=x, y=y)

    @staticmethod
    def swipe(t: float, x: int, y: int, x2: int, y2: int) -> "Action":
        return Action(type=ActionType.SWIPE, t=t, x=x, y=y, x2=x2, y2=y2)
