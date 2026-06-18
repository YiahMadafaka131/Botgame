"""Closed-loop adversarial reward + row-based detector scorers."""

from __future__ import annotations

import numpy as np
import pytest

from botgame.dataset.schema import Action
from botgame.humanize import Humanizer
from botgame.redteam.detectors import (
    composite_score,
    coord_cluster_score,
    default_composite,
    default_composite_score,
    perfect_aim_score,
    periodicity_detector,
    periodicity_score,
    reaction_time_score,
)
from botgame.rl import compose_rewards, detection_evasion_reward
from botgame.rl.rewards import pixel_diff_reward


def _frame(value: int = 0, shape=(20, 20, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


# ---- row-based scorers match path-based detectors -------------------


def test_row_scorers_match_path_detectors(tmp_path):
    import json
    rows = [
        {"ts": i * 0.5, "action": "tap", "reaction_s": 0.0,
         "target": [500, 500], "actual": [500, 500]}
        for i in range(10)
    ]
    path = tmp_path / "robotic.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    assert periodicity_score(rows) == pytest.approx(periodicity_detector(str(path)))
    assert default_composite_score(rows) == pytest.approx(default_composite(str(path)))


# ---- detection_evasion_reward --------------------------------------


def _fake_clock():
    """Steady metronomic timestamps — periodicity_score sees CV ≈ 0."""
    t = [0.0]

    def _next() -> float:
        t[0] += 0.1
        return t[0]

    return _next


def test_returns_zero_before_min_actions():
    r = detection_evasion_reward(periodicity_score, min_actions=5,
                                 clock=_fake_clock())
    # 4 taps < min_actions=5.
    for i in range(4):
        out = r(_frame(), Action.tap(0.0, x=100, y=100), _frame())
        assert out == 0.0


def test_penalises_metronomic_stream():
    r = detection_evasion_reward(periodicity_score, scale=1.0,
                                 min_actions=3, clock=_fake_clock())
    out = 0.0
    for _ in range(10):
        out = r(_frame(), Action.tap(0.0, x=100, y=100), _frame())
    # Constant interval → periodicity ≈ 1 → reward ≈ -1.
    assert out < -0.9


def test_penalises_coord_clustering():
    r = detection_evasion_reward(coord_cluster_score, scale=1.0,
                                 min_actions=3, clock=_fake_clock())
    out = 0.0
    for _ in range(10):
        out = r(_frame(), Action.tap(0.0, x=500, y=500), _frame())
    assert out == pytest.approx(-1.0)


def test_reward_rises_when_policy_diversifies():
    r = detection_evasion_reward(coord_cluster_score, scale=1.0,
                                 min_actions=3, clock=_fake_clock())
    # Clustered first — strong penalty.
    for _ in range(10):
        clustered = r(_frame(), Action.tap(0.0, x=500, y=500), _frame())
    # Then dispersed — penalty fades as buffer rotates in spread coords.
    for i in range(20):
        dispersed = r(_frame(), Action.tap(0.0, x=100 + i * 40, y=100 + i * 37),
                      _frame())
    assert dispersed > clustered + 0.5


def test_perfect_aim_score_full_when_actual_equals_target():
    r = detection_evasion_reward(perfect_aim_score, scale=1.0,
                                 min_actions=3, clock=_fake_clock())
    out = 0.0
    for _ in range(5):
        out = r(_frame(), Action.tap(0.0, x=10, y=20), _frame())
    # actual == target in every row → perfect_aim_score == 1.0 → reward = -1.
    assert out == pytest.approx(-1.0)


def test_reaction_time_score_high_when_no_humanizer():
    r = detection_evasion_reward(reaction_time_score, scale=1.0,
                                 min_actions=3, clock=_fake_clock())
    out = 0.0
    for _ in range(5):
        out = r(_frame(), Action.tap(0.0, x=10, y=20), _frame())
    # reaction_s defaults to 0 in row builder → 100% sub-human → reward = -1.
    assert out == pytest.approx(-1.0)


def test_scale_multiplies_penalty():
    r = detection_evasion_reward(perfect_aim_score, scale=0.25,
                                 min_actions=3, clock=_fake_clock())
    out = 0.0
    for _ in range(5):
        out = r(_frame(), Action.tap(0.0, x=10, y=20), _frame())
    assert out == pytest.approx(-0.25)


def test_buffer_size_limits_window():
    r = detection_evasion_reward(coord_cluster_score, scale=1.0,
                                 buffer_size=4, min_actions=3,
                                 clock=_fake_clock())
    # Feed 3 clustered then 4 dispersed — buffer holds last 4 → mostly dispersed.
    for _ in range(3):
        r(_frame(), Action.tap(0.0, x=500, y=500), _frame())
    last = 0.0
    for i in range(4):
        last = r(_frame(), Action.tap(0.0, x=100 + i * 50, y=200 + i * 40),
                 _frame())
    # Buffer fully rotated → reward closer to 0 than -1.
    assert last > -0.5


def test_noop_actions_dont_break_scoring():
    r = detection_evasion_reward(periodicity_score, scale=1.0,
                                 min_actions=3, clock=_fake_clock())
    for _ in range(5):
        out = r(_frame(), Action.noop(0.0), _frame())
    # noop rows are filtered out by _action_records → score 0 → reward 0.
    assert out == 0.0


def test_rejects_zero_buffer_size():
    with pytest.raises(ValueError):
        detection_evasion_reward(periodicity_score, buffer_size=0)


def test_composes_with_other_rewards():
    adversarial = detection_evasion_reward(perfect_aim_score, scale=2.0,
                                           min_actions=3, clock=_fake_clock())
    reward = compose_rewards([
        (pixel_diff_reward(), 1.0),
        (adversarial, 1.0),
    ])
    out = 0.0
    for _ in range(5):
        out = reward(_frame(0), Action.tap(0.0, x=10, y=20), _frame(255))
    # pixel_diff saturates at +1; adversarial = -2 → net ≈ -1 → composition active.
    assert out == pytest.approx(-1.0)


def test_supports_swipe_actions():
    r = detection_evasion_reward(periodicity_score, min_actions=3,
                                 clock=_fake_clock())
    out = 0.0
    for _ in range(5):
        out = r(_frame(), Action.swipe(0.0, x=10, y=10, x2=200, y2=200),
                _frame())
    # Swipes are kept by _action_records → periodicity fires on steady clock.
    assert out < -0.9


def test_composite_score_validates_inputs():
    with pytest.raises(ValueError):
        composite_score([])
    with pytest.raises(ValueError):
        composite_score([periodicity_score], weights=[0.5, 0.5])
    with pytest.raises(ValueError):
        composite_score([periodicity_score], weights=[0.0])


def test_composite_score_custom_weights():
    rows = [
        {"ts": i * 0.5, "action": "tap", "reaction_s": 0.0,
         "target": [500, 500], "actual": [500, 500]}
        for i in range(10)
    ]
    # Weight 1.0 on a single scorer reduces to that scorer.
    scorer = composite_score([periodicity_score], weights=[1.0])
    assert scorer(rows) == pytest.approx(periodicity_score(rows))


# ---- humanizer integration -----------------------------------------


def test_humanizer_disperses_actual_coords():
    """With humanizer, tap rows carry jittered actual coords."""
    h = Humanizer(level=1.0, seed=42)
    r = detection_evasion_reward(coord_cluster_score, scale=1.0,
                                 min_actions=3, humanizer=h,
                                 clock=_fake_clock())
    out = 0.0
    for _ in range(20):
        out = r(_frame(), Action.tap(0.0, x=500, y=500), _frame())
    # Same target every step → without humanizer, cluster ≈ 1, reward ≈ -1.
    # With humanizer level=1 → sigma ≈ 12 px → coord_cluster_score drops.
    assert out > -0.5


def test_humanizer_breaks_perfect_aim_saturation():
    """perfect_aim_score should be sub-1 once humanizer jitters coords."""
    h = Humanizer(level=1.0, seed=0)
    r = detection_evasion_reward(perfect_aim_score, scale=1.0,
                                 min_actions=3, humanizer=h,
                                 clock=_fake_clock())
    out = 0.0
    for _ in range(20):
        out = r(_frame(), Action.tap(0.0, x=500, y=500), _frame())
    # Jittered actual != target most of the time → score well below 1.
    assert out > -0.5


def test_humanizer_breaks_reaction_time_saturation():
    """reaction_time_score drops once humanizer samples real delays."""
    h = Humanizer(level=1.0, seed=0)
    r = detection_evasion_reward(reaction_time_score, scale=1.0,
                                 min_actions=3, humanizer=h,
                                 clock=_fake_clock())
    out = 0.0
    for _ in range(20):
        out = r(_frame(), Action.tap(0.0, x=500, y=500), _frame())
    # base_reaction_s=0.18 → mostly above 80ms cutoff → score → 0.
    assert out > -0.2


def test_humanizer_level_zero_matches_no_humanizer():
    """Humanizer(level=0) should behave identically to no humanizer."""
    h = Humanizer(level=0.0, seed=0)
    r_with = detection_evasion_reward(perfect_aim_score, scale=1.0,
                                      min_actions=3, humanizer=h,
                                      clock=_fake_clock())
    r_without = detection_evasion_reward(perfect_aim_score, scale=1.0,
                                         min_actions=3, clock=_fake_clock())
    out_with, out_without = 0.0, 0.0
    for _ in range(5):
        out_with = r_with(_frame(), Action.tap(0.0, x=10, y=20), _frame())
        out_without = r_without(_frame(), Action.tap(0.0, x=10, y=20), _frame())
    assert out_with == pytest.approx(out_without)


# ---- PPO integration smoke -----------------------------------------


def test_ppo_runs_with_adversarial_reward_end_to_end():
    """Smoke: PPO trains a few steps with a composed adversarial reward."""
    pytest.importorskip("torch")
    from botgame.rl import BotEnv, PPOConfig, PPOTrainer

    class FakeCap:
        def __init__(self):
            self.rng = np.random.default_rng(0)

        def grab(self):
            return self.rng.integers(0, 256, size=(22, 40, 3), dtype=np.uint8)

    class FakeTouch:
        def tap(self, x, y): pass
        def swipe_path(self, points, duration_ms=300): pass

    adversarial = detection_evasion_reward(
        coord_cluster_score, scale=0.5,
        min_actions=3, humanizer=Humanizer(level=0.5, seed=0),
        clock=_fake_clock(),
    )
    reward = compose_rewards([
        (pixel_diff_reward(), 1.0),
        (adversarial, 1.0),
    ])

    env = BotEnv(FakeCap(), FakeTouch(), reward_fn=reward,
                 step_interval_s=0.0, max_steps=8)
    cfg = PPOConfig(
        rollout_steps=8, epochs=1, minibatch_size=4,
        screen_size=(40, 22), input_size=(40, 22),
    )
    trainer = PPOTrainer(env, config=cfg)
    history = trainer.train(total_steps=8)
    assert len(history) == 1
    assert np.isfinite(history[0].policy_loss)
