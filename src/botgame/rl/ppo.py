"""Small PPO trainer for fine-tuning the imitation-learned policy.

This is deliberately a compact, single-file PPO — enough to run a few thousand
steps on a CPU with the included `RandomEnv` and verify the actor + critic
both learn. For serious training swap in stable-baselines3, but the env / net
interfaces here are designed so the rest of the pipeline doesn't notice.

Action sampling at rollout time:
  - type ~ Categorical(softmax(type_logits))
  - coords ~ Normal(coord_mean, exp(log_sigma)) then clamp to [0, 1]

PPO loss = clipped policy ratio + value MSE + entropy bonus.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..dataset.schema import Action, ActionType
from ..model.encoding import ACTION_TYPES, decode_prediction
from .env import EnvStep
from .net import build_actor_critic, load_bc_weights


@dataclass
class PPOConfig:
    rollout_steps: int = 64
    epochs: int = 4
    minibatch_size: int = 32
    lr: float = 3e-4
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    screen_size: tuple[int, int] = (1080, 2400)
    input_size: tuple[int, int] = (160, 90)  # (W, H) — must match BC training


@dataclass
class RolloutStats:
    mean_reward: float
    mean_value: float
    policy_loss: float
    value_loss: float
    entropy: float
    elapsed_s: float = 0.0
    epoch_index: int = 0
    extras: dict = field(default_factory=dict)


def _resize_nn(frame_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    sh, sw = frame_rgb.shape[:2]
    ys = np.linspace(0, sh - 1, h).astype(np.intp)
    xs = np.linspace(0, sw - 1, w).astype(np.intp)
    return frame_rgb[ys][:, xs]


class PPOTrainer:
    def __init__(
        self,
        env,
        config: PPOConfig | None = None,
        bc_checkpoint: str | None = None,
        device: str | None = None,
    ):
        import torch

        self.torch = torch
        self.env = env
        self.config = config or PPOConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_actor_critic().to(self.device)
        if bc_checkpoint:
            load_bc_weights(self.net, bc_checkpoint)
            self.net.to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.config.lr)

    # ---- forward / sampling -------------------------------------------

    def _preprocess(self, frame: np.ndarray):
        torch = self.torch
        small = _resize_nn(frame, self.config.input_size)
        t = torch.from_numpy(small).float().permute(2, 0, 1) / 255.0
        return t.unsqueeze(0).to(self.device)

    def _sample(self, frame: np.ndarray):
        torch = self.torch
        x = self._preprocess(frame)
        type_logits, coord_mean, value = self.net(x)
        type_dist = torch.distributions.Categorical(logits=type_logits)
        sigma = torch.exp(self.net.log_sigma).clamp(min=1e-3, max=1.0)
        coord_dist = torch.distributions.Normal(coord_mean, sigma)
        type_idx = type_dist.sample()
        coords_raw = coord_dist.sample()
        coords = coords_raw.clamp(0.0, 1.0)
        logp = type_dist.log_prob(type_idx) + coord_dist.log_prob(coords_raw).sum(-1)
        return (
            int(type_idx.item()),
            coords.squeeze(0).detach().cpu().numpy().tolist(),
            float(logp.item()),
            float(value.item()),
        )

    def _action_from(self, type_idx: int, coords: list[float]) -> Action:
        w, h = self.config.screen_size
        return decode_prediction(type_idx, coords, w, h)

    # ---- rollout ------------------------------------------------------

    def collect_rollout(self):
        torch = self.torch
        cfg = self.config
        frames: list[np.ndarray] = []
        type_idxs: list[int] = []
        coords_list: list[list[float]] = []
        logps: list[float] = []
        values: list[float] = []
        rewards: list[float] = []
        dones: list[bool] = []

        frame = self.env.reset()
        for _ in range(cfg.rollout_steps):
            type_idx, coords, logp, value = self._sample(frame)
            action = self._action_from(type_idx, coords)
            step: EnvStep = self.env.step(action)
            frames.append(frame)
            type_idxs.append(type_idx)
            coords_list.append(coords)
            logps.append(logp)
            values.append(value)
            rewards.append(step.reward)
            dones.append(step.done)
            frame = step.frame if not step.done else self.env.reset()

        # Bootstrap with V(s_T).
        with torch.no_grad():
            _, _, last_value = self.net(self._preprocess(frame))
        last_value = float(last_value.item())

        advantages, returns = self._gae(rewards, values, dones, last_value)
        return {
            "frames": frames,
            "type_idxs": type_idxs,
            "coords": coords_list,
            "logps": logps,
            "values": values,
            "rewards": rewards,
            "dones": dones,
            "advantages": advantages,
            "returns": returns,
        }

    def _gae(self, rewards, values, dones, last_value) -> tuple[list[float], list[float]]:
        cfg = self.config
        n = len(rewards)
        advantages = [0.0] * n
        gae = 0.0
        for t in reversed(range(n)):
            next_v = last_value if t == n - 1 else values[t + 1]
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + cfg.gamma * next_v * mask - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * mask * gae
            advantages[t] = gae
        returns = [a + v for a, v in zip(advantages, values)]
        return advantages, returns

    # ---- update -------------------------------------------------------

    def update(self, rollout) -> RolloutStats:
        torch = self.torch
        cfg = self.config
        device = self.device

        x = torch.stack(
            [self._preprocess(f).squeeze(0) for f in rollout["frames"]], dim=0
        )
        type_idxs = torch.tensor(rollout["type_idxs"], dtype=torch.long, device=device)
        coords = torch.tensor(rollout["coords"], dtype=torch.float32, device=device)
        old_logps = torch.tensor(rollout["logps"], dtype=torch.float32, device=device)
        returns = torch.tensor(rollout["returns"], dtype=torch.float32, device=device)
        advantages = torch.tensor(
            rollout["advantages"], dtype=torch.float32, device=device
        )
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = x.size(0)
        idxs = np.arange(n)
        last_policy_loss = last_value_loss = last_entropy = 0.0
        for _ in range(cfg.epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, cfg.minibatch_size):
                mb = idxs[start:start + cfg.minibatch_size]
                mb_t = torch.from_numpy(mb).to(device)
                xb = x.index_select(0, mb_t)
                type_logits, coord_mean, value_pred = self.net(xb)

                type_dist = torch.distributions.Categorical(logits=type_logits)
                sigma = torch.exp(self.net.log_sigma).clamp(min=1e-3, max=1.0)
                coord_dist = torch.distributions.Normal(coord_mean, sigma)

                new_logp = (
                    type_dist.log_prob(type_idxs[mb_t])
                    + coord_dist.log_prob(coords[mb_t]).sum(-1)
                )
                ratio = torch.exp(new_logp - old_logps[mb_t])
                adv = advantages[mb_t]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((value_pred - returns[mb_t]) ** 2).mean()
                entropy = type_dist.entropy().mean() + coord_dist.entropy().sum(-1).mean()
                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy
                )
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optim.step()
                last_policy_loss = float(policy_loss.item())
                last_value_loss = float(value_loss.item())
                last_entropy = float(entropy.item())

        return RolloutStats(
            mean_reward=float(np.mean(rollout["rewards"])),
            mean_value=float(np.mean(rollout["values"])),
            policy_loss=last_policy_loss,
            value_loss=last_value_loss,
            entropy=last_entropy,
        )

    # ---- top-level ----------------------------------------------------

    def train(
        self,
        total_steps: int,
        on_iter: Callable[[RolloutStats], None] | None = None,
    ) -> list[RolloutStats]:
        cfg = self.config
        n_iters = max(1, total_steps // cfg.rollout_steps)
        history: list[RolloutStats] = []
        for i in range(n_iters):
            t0 = time.monotonic()
            rollout = self.collect_rollout()
            stats = self.update(rollout)
            stats.elapsed_s = time.monotonic() - t0
            stats.epoch_index = i
            history.append(stats)
            if on_iter is not None:
                on_iter(stats)
        return history

    def save(self, path: str) -> None:
        self.torch.save({"state_dict": self.net.state_dict()}, path)


def train_rl(
    env,
    out_path: str,
    total_steps: int,
    bc_checkpoint: str | None = None,
    config: PPOConfig | None = None,
) -> list[RolloutStats]:
    trainer = PPOTrainer(env, config=config, bc_checkpoint=bc_checkpoint)
    history = trainer.train(total_steps)
    trainer.save(out_path)
    return history
