"""PPO-trainer smoke tests against the stub RandomEnv (no device required)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from botgame.dataset.schema import Action, ActionType
from botgame.rl import BotEnv, PPOConfig, PPOTrainer, RandomEnv


def test_random_env_emits_frames_and_terminates():
    env = RandomEnv(max_steps=3, seed=0)
    f = env.reset()
    assert f.shape == (90, 160, 3)
    assert f.dtype == np.uint8
    rewards = []
    done = False
    while not done:
        step = env.step(Action.tap(0.0, 80, 45))
        rewards.append(step.reward)
        done = step.done
    assert len(rewards) == 3


def test_random_env_rewards_central_taps_more_than_edges():
    env = RandomEnv(seed=1)
    env.reset()
    centre = env.step(Action.tap(0.0, 80, 45)).reward
    env.reset()
    edge = env.step(Action.tap(0.0, 0, 0)).reward
    assert centre > edge


def test_ppo_one_iteration_runs_and_returns_stats():
    cfg = PPOConfig(
        rollout_steps=8, epochs=2, minibatch_size=4,
        screen_size=(160, 90), input_size=(40, 22),
    )
    trainer = PPOTrainer(RandomEnv(max_steps=4, seed=2), config=cfg)
    history = trainer.train(total_steps=16)
    assert len(history) == 2
    for s in history:
        assert np.isfinite(s.policy_loss)
        assert np.isfinite(s.value_loss)
        assert np.isfinite(s.entropy)


def test_ppo_updates_parameters():
    cfg = PPOConfig(
        rollout_steps=8, epochs=2, minibatch_size=4,
        screen_size=(160, 90), input_size=(40, 22),
    )
    trainer = PPOTrainer(RandomEnv(seed=3), config=cfg)
    before = [p.detach().clone() for p in trainer.net.parameters()]
    trainer.train(total_steps=8)
    after = list(trainer.net.parameters())
    moved = sum(1 for a, b in zip(after, before) if not torch.allclose(a, b))
    assert moved > 0, "PPO should have updated at least some parameters"


def test_ppo_save_and_reload(tmp_path):
    cfg = PPOConfig(
        rollout_steps=4, epochs=1, minibatch_size=4,
        input_size=(40, 22), screen_size=(160, 90),
    )
    trainer = PPOTrainer(RandomEnv(seed=4), config=cfg)
    trainer.train(total_steps=4)
    out = tmp_path / "rl.pt"
    trainer.save(str(out))
    assert out.exists()
    loaded = torch.load(str(out), map_location="cpu", weights_only=False)
    assert "state_dict" in loaded


def test_botenv_step_calls_touch_and_capture():
    """Live env wires capture + touch + reward together."""

    grabs: list[int] = []
    taps: list[tuple[int, int]] = []

    class FakeCap:
        def grab(self):
            grabs.append(1)
            return np.zeros((4, 4, 3), dtype=np.uint8)

    class FakeTouch:
        def tap(self, x, y):
            taps.append((x, y))

        def swipe_path(self, points, duration_ms=300):
            taps.extend(points)

    env = BotEnv(
        FakeCap(), FakeTouch(),
        reward_fn=lambda prev, action, nxt: 1.0,
        step_interval_s=0.0,
        max_steps=2,
    )
    env.reset()
    s = env.step(Action.tap(0.0, 5, 7))
    assert taps == [(5, 7)]
    assert s.reward == 1.0
    assert not s.done
    s2 = env.step(Action.noop(0.0))
    assert s2.done  # max_steps reached


def test_botenv_step_requires_reset_first():
    env = BotEnv(
        capture=type("C", (), {"grab": lambda self: np.zeros((1, 1, 3), dtype=np.uint8)})(),
        touch=type("T", (), {
            "tap": lambda self, x, y: None,
            "swipe_path": lambda self, p, duration_ms=300: None,
        })(),
        reward_fn=lambda *_: 0.0,
    )
    with pytest.raises(RuntimeError):
        env.step(Action.noop(0.0))
