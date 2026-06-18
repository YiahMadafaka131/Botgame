"""Actor-critic head used by PPO fine-tuning (lazy torch import).

Same trunk as the behavioral-cloning policy so a BC checkpoint loads cleanly
via `load_bc_weights`. Adds:
  - a value head (scalar baseline V(s) per frame),
  - a learnable log-sigma for the coordinate distribution (treated as a
    diagonal Gaussian during rollouts; clipped to [0, 1] before dispatch).
"""

from __future__ import annotations


def build_actor_critic(n_types: int = 3):
    import torch
    import torch.nn as nn

    class ActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            )
            self.type_head = nn.Linear(128, n_types)
            self.coord_head = nn.Sequential(nn.Linear(128, 4), nn.Sigmoid())
            self.value_head = nn.Linear(128, 1)
            self.log_sigma = nn.Parameter(torch.full((4,), -1.5))

        def forward(self, x):
            z = self.head(self.features(x))
            return (
                self.type_head(z),
                self.coord_head(z),
                self.value_head(z).squeeze(-1),
            )

    return ActorCritic()


def load_bc_weights(actor_critic, bc_checkpoint_path: str) -> None:
    """Copy matching weights from a BC PolicyNet checkpoint into the actor-critic.

    Skips keys the actor-critic does not have (e.g. nothing for value_head /
    log_sigma — they keep their initial values).
    """
    import torch

    ckpt = torch.load(bc_checkpoint_path, map_location="cpu")
    src = ckpt.get("state_dict", ckpt)
    own = actor_critic.state_dict()
    loaded = 0
    for k, v in src.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
            loaded += 1
    actor_critic.load_state_dict(own)
    return loaded
