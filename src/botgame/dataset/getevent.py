"""Parse `adb shell getevent -lt` output into touch Actions.

This is the most reliable way to recover ground-truth actions: while you record
gameplay, also capture `getevent -lt` and you get timestamped, pixel-accurate
touch events independent of any on-screen overlay.

Relevant lines look like:

    [   12345.678901] /dev/input/event3: EV_ABS  ABS_MT_POSITION_X  000004a3
    [   12345.678901] /dev/input/event3: EV_ABS  ABS_MT_POSITION_Y  00000b1c
    [   12345.678901] /dev/input/event3: EV_KEY  BTN_TOUCH          DOWN
    [   12345.690000] /dev/input/event3: EV_SYN  SYN_REPORT         00000000
    [   12345.789000] /dev/input/event3: EV_KEY  BTN_TOUCH          UP

Coordinates are in the touch device's own space (hex). Pass `src_size` and
`dst_size` to rescale them to screen pixels; otherwise raw values are kept.
"""

from __future__ import annotations

import re

from .schema import Action

_LINE = re.compile(
    r"\[\s*(?P<t>\d+\.\d+)\]\s+\S+:\s+(?P<etype>\S+)\s+(?P<code>\S+)\s+(?P<val>\S+)"
)


def _scale(v: int, src: int | None, dst: int | None) -> int:
    if src and dst and src > 0:
        return int(round(v * dst / src))
    return v


def parse_getevent(
    text: str,
    *,
    src_size: tuple[int, int] | None = None,
    dst_size: tuple[int, int] | None = None,
    swipe_threshold_px: int = 25,
    t0: float | None = None,
) -> list[Action]:
    """Convert getevent text into a time-ordered list of TAP/SWIPE Actions.

    A contact from BTN_TOUCH DOWN to UP becomes a TAP if the finger barely
    moved, else a SWIPE. Times are made relative to the first event (or `t0`).
    """
    sx = src_size[0] if src_size else None
    sy = src_size[1] if src_size else None
    dx = dst_size[0] if dst_size else None
    dy = dst_size[1] if dst_size else None

    actions: list[Action] = []
    cur_x: int | None = None
    cur_y: int | None = None
    down_x: int | None = None
    down_y: int | None = None
    down_t: float | None = None
    contact = False

    for line in text.splitlines():
        m = _LINE.search(line)
        if not m:
            continue
        t = float(m.group("t"))
        if t0 is None:
            t0 = t
        code, val = m.group("code"), m.group("val")

        if code == "ABS_MT_POSITION_X":
            cur_x = _scale(int(val, 16), sx, dx)
        elif code == "ABS_MT_POSITION_Y":
            cur_y = _scale(int(val, 16), sy, dy)
        elif code == "BTN_TOUCH":
            if val == "DOWN":
                contact = True
                down_x, down_y, down_t = cur_x, cur_y, t
            elif val == "UP" and contact:
                contact = False
                ux, uy = cur_x, cur_y
                rel_t = (down_t if down_t is not None else t) - t0
                if down_x is None or down_y is None:
                    continue
                ex = ux if ux is not None else down_x
                ey = uy if uy is not None else down_y
                dist = max(abs(ex - down_x), abs(ey - down_y))
                if dist <= swipe_threshold_px:
                    actions.append(Action.tap(rel_t, down_x, down_y))
                else:
                    actions.append(Action.swipe(rel_t, down_x, down_y, ex, ey))

    actions.sort(key=lambda a: a.t)
    return actions
