import numpy as np

from botgame.dataset import parse_getevent, detect_touch, align_actions
from botgame.dataset.schema import Action, ActionType


GETEVENT_SAMPLE = """\
[   100.000000] /dev/input/event3: EV_ABS       ABS_MT_POSITION_X    00000064
[   100.000000] /dev/input/event3: EV_ABS       ABS_MT_POSITION_Y    000000c8
[   100.000000] /dev/input/event3: EV_KEY       BTN_TOUCH            DOWN
[   100.010000] /dev/input/event3: EV_SYN       SYN_REPORT           00000000
[   100.050000] /dev/input/event3: EV_KEY       BTN_TOUCH            UP
[   100.050000] /dev/input/event3: EV_SYN       SYN_REPORT           00000000
[   101.000000] /dev/input/event3: EV_ABS       ABS_MT_POSITION_X    00000064
[   101.000000] /dev/input/event3: EV_ABS       ABS_MT_POSITION_Y    000000c8
[   101.000000] /dev/input/event3: EV_KEY       BTN_TOUCH            DOWN
[   101.200000] /dev/input/event3: EV_ABS       ABS_MT_POSITION_X    00000320
[   101.200000] /dev/input/event3: EV_ABS       ABS_MT_POSITION_Y    000000c8
[   101.250000] /dev/input/event3: EV_KEY       BTN_TOUCH            UP
"""


def test_parse_getevent_tap_and_swipe():
    actions = parse_getevent(GETEVENT_SAMPLE)
    assert len(actions) == 2

    tap = actions[0]
    assert tap.type is ActionType.TAP
    assert (tap.x, tap.y) == (0x64, 0xC8)
    assert tap.t == 0.0  # first event is t0

    swipe = actions[1]
    assert swipe.type is ActionType.SWIPE
    assert (swipe.x, swipe.y) == (0x64, 0xC8)
    assert (swipe.x2, swipe.y2) == (0x320, 0xC8)
    assert abs(swipe.t - 1.0) < 1e-9


def test_parse_getevent_rescales_coords():
    actions = parse_getevent(
        GETEVENT_SAMPLE, src_size=(1000, 1000), dst_size=(500, 2000)
    )
    tap = actions[0]
    # x: 100/1000*500 = 50 ; y: 200/1000*2000 = 400
    assert (tap.x, tap.y) == (50, 400)


def test_detect_touch_finds_bright_blob():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[40:60, 90:110] = 255  # bright square centred at (~99, ~49)
    hit = detect_touch(frame)
    assert hit is not None
    x, y = hit
    assert abs(x - 99) <= 2
    assert abs(y - 49) <= 2


def test_detect_touch_returns_none_on_dark_frame():
    frame = np.full((100, 200, 3), 10, dtype=np.uint8)
    assert detect_touch(frame) is None


def test_align_actions_windows():
    actions = [Action.tap(0.5, 10, 10), Action.swipe(2.05, 0, 0, 9, 9)]
    frame_times = [0.0, 0.5, 1.0, 2.0]  # fps=2 -> window 0.5s
    labels = align_actions(frame_times, actions, fps=2.0)
    assert [label.type for label in labels] == [
        ActionType.NOOP,   # 0.0..0.5 -> tap at 0.5 is excluded (half-open)
        ActionType.TAP,    # 0.5..1.0 -> tap@0.5
        ActionType.NOOP,
        ActionType.SWIPE,  # 2.0..2.5 -> swipe@2.05
    ]


def test_align_actions_rejects_bad_fps():
    import pytest

    with pytest.raises(ValueError):
        align_actions([0.0], [], fps=0)
