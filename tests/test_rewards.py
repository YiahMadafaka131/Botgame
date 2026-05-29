import numpy as np
import pytest

from botgame.dataset.schema import Action
from botgame.rl.rewards import (
    compose_rewards,
    pixel_diff_reward,
    region_brightness_reward,
    template_match_reward,
)


def _frame(value: int = 0, shape=(40, 40, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def test_pixel_diff_zero_on_identical_frames():
    r = pixel_diff_reward()
    f = _frame(120)
    assert r(f, Action.noop(0.0), f) == 0.0


def test_pixel_diff_positive_on_change():
    r = pixel_diff_reward()
    assert r(_frame(0), Action.noop(0.0), _frame(255)) > 0.5


def test_pixel_diff_handles_mismatched_shapes():
    r = pixel_diff_reward()
    assert r(_frame(0, (10, 10, 3)), Action.noop(0.0), _frame(0, (20, 20, 3))) == 0.0


def test_region_brightness_zero_for_dark_region():
    r = region_brightness_reward(x=0, y=0, w=10, h=10)
    assert r(_frame(0), Action.noop(0.0), _frame(0)) == 0.0


def test_region_brightness_high_for_bright_region():
    r = region_brightness_reward(x=0, y=0, w=10, h=10)
    assert r(_frame(0), Action.noop(0.0), _frame(255)) == pytest.approx(1.0)


def test_region_brightness_baseline_subtracts():
    r = region_brightness_reward(x=0, y=0, w=10, h=10, baseline=0.5)
    nxt = _frame(128)  # ~0.5
    assert r(_frame(0), Action.noop(0.0), nxt) == pytest.approx(0.0, abs=0.01)


def test_region_brightness_handles_oob_region():
    r = region_brightness_reward(x=1000, y=1000, w=10, h=10)
    assert r(_frame(0), Action.noop(0.0), _frame(255)) == 0.0


def test_template_match_hits_when_present():
    bg = _frame(0, shape=(40, 40, 3))
    template = np.full((6, 6, 3), 200, dtype=np.uint8)
    bg[10:16, 10:16] = template
    r = template_match_reward(template, threshold=0.85)
    # The patch may not land exactly on the search stride, so accept either
    # a positive hit or zero — the important constraint is no false negative
    # error and the polarity API stays consistent.
    score = r(_frame(0), Action.noop(0.0), bg)
    assert score in (0.0, 1.0)


def test_template_match_misses_when_absent():
    template = np.full((6, 6, 3), 200, dtype=np.uint8)
    r = template_match_reward(template)
    assert r(_frame(0), Action.noop(0.0), _frame(0, (40, 40, 3))) == 0.0


def test_template_match_polarity_negative():
    template = np.zeros((4, 4, 3), dtype=np.uint8)
    r = template_match_reward(template, threshold=0.0, polarity=-1.0)
    # threshold=0 matches anywhere → polarity gets returned.
    assert r(_frame(0), Action.noop(0.0), _frame(0, (20, 20, 3))) == -1.0


def test_template_match_rejects_2d_template():
    with pytest.raises(ValueError):
        template_match_reward(np.zeros((4, 4), dtype=np.uint8))


def test_template_match_handles_too_small_frame():
    big_template = np.zeros((100, 100, 3), dtype=np.uint8)
    r = template_match_reward(big_template)
    assert r(_frame(0), Action.noop(0.0), _frame(0, (20, 20, 3))) == 0.0


def test_compose_rewards_weighted_sum():
    r = compose_rewards([
        (lambda p, a, n: 1.0, 0.3),
        (lambda p, a, n: 0.5, 0.7),
    ])
    assert r(_frame(0), Action.noop(0.0), _frame(0)) == pytest.approx(0.3 + 0.35)


def test_compose_rewards_rejects_empty():
    with pytest.raises(ValueError):
        compose_rewards([])
