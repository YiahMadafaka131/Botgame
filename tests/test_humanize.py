import numpy as np
import pytest

from botgame.humanize import Humanizer


def test_level_zero_is_deterministic_and_robotic():
    h = Humanizer(level=0.0)
    assert h.jitter_point(100, 200) == (100, 200)
    assert h.reaction_delay() == 0.0
    assert h.action_interval(0.5) == 0.5


def test_level_zero_bezier_is_straight_line():
    h = Humanizer(level=0.0)
    path = h.bezier_path((0, 0), (100, 0), n_points=5)
    # All y should be 0 (straight horizontal line), evenly spaced x.
    assert [p[1] for p in path] == [0, 0, 0, 0, 0]
    assert [p[0] for p in path] == [0, 25, 50, 75, 100]


def test_endpoints_are_pinned_even_when_human():
    h = Humanizer(level=1.0, seed=1)
    path = h.bezier_path((10, 20), (300, 400), n_points=12)
    assert path[0] == (10, 20)
    assert path[-1] == (300, 400)


def test_jitter_scales_with_level():
    pts0 = [Humanizer(level=0.0, seed=s).jitter_point(500, 500) for s in range(200)]
    pts1 = [Humanizer(level=1.0, seed=s).jitter_point(500, 500) for s in range(200)]
    spread0 = np.std([p[0] for p in pts0])
    spread1 = np.std([p[0] for p in pts1])
    assert spread0 == 0.0
    assert spread1 > spread0


def test_reaction_delay_positive_when_human():
    h = Humanizer(level=1.0, seed=7)
    delays = [h.reaction_delay() for _ in range(100)]
    assert all(d > 0 for d in delays)
    assert np.mean(delays) > 0.1


def test_action_interval_never_negative():
    h = Humanizer(level=1.0, seed=3)
    vals = [h.action_interval(0.05) for _ in range(500)]
    assert all(v >= 0 for v in vals)


def test_invalid_level_rejected():
    with pytest.raises(ValueError):
        Humanizer(level=1.5)
    with pytest.raises(ValueError):
        Humanizer(level=-0.1)


def test_seed_reproducibility():
    a = Humanizer(level=0.8, seed=42)
    b = Humanizer(level=0.8, seed=42)
    assert a.jitter_point(100, 100) == b.jitter_point(100, 100)
    assert a.bezier_path((0, 0), (50, 50)) == b.bezier_path((0, 0), (50, 50))
