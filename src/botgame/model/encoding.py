"""Convert between Actions and model targets (pure, no torch).

The policy head predicts:
  - an action *type* (noop / tap / swipe) as a 3-way classification, and
  - four normalized coordinates (x, y, x2, y2) in [0, 1] as regression.

Keeping this conversion separate and dependency-free makes it unit-testable and
shared by both training (label -> target) and inference (output -> Action).
"""

from __future__ import annotations

from ..dataset.schema import Action, ActionType

# Fixed class order; the model's output index maps to this list.
ACTION_TYPES: tuple[ActionType, ...] = (ActionType.NOOP, ActionType.TAP, ActionType.SWIPE)
TYPE_TO_INDEX = {t: i for i, t in enumerate(ACTION_TYPES)}


def action_to_target(action: Action, width: int, height: int) -> tuple[int, list[float]]:
    """Return (type_index, [x, y, x2, y2]) with coords normalized to [0, 1]."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    idx = TYPE_TO_INDEX[action.type]

    def nx(v: int | None) -> float:
        return 0.0 if v is None else min(1.0, max(0.0, v / width))

    def ny(v: int | None) -> float:
        return 0.0 if v is None else min(1.0, max(0.0, v / height))

    if action.type is ActionType.TAP:
        coords = [nx(action.x), ny(action.y), nx(action.x), ny(action.y)]
    elif action.type is ActionType.SWIPE:
        coords = [nx(action.x), ny(action.y), nx(action.x2), ny(action.y2)]
    else:  # NOOP
        coords = [0.0, 0.0, 0.0, 0.0]
    return idx, coords


def decode_prediction(
    type_index: int,
    coords: list[float],
    width: int,
    height: int,
    t: float = 0.0,
) -> Action:
    """Turn a model output (type index + normalized coords) into an Action."""
    atype = ACTION_TYPES[type_index]
    px = int(round(coords[0] * width))
    py = int(round(coords[1] * height))
    px2 = int(round(coords[2] * width))
    py2 = int(round(coords[3] * height))

    if atype is ActionType.TAP:
        return Action.tap(t, px, py)
    if atype is ActionType.SWIPE:
        return Action.swipe(t, px, py, px2, py2)
    return Action.noop(t)
