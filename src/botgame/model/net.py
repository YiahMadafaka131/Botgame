"""The policy network (lazy torch import).

A small CNN that maps a downscaled RGB frame to an action type (3-way) and four
normalized coordinates. Deliberately tiny so it trains fast on a single GPU/CPU
and runs in real time during play.
"""

from __future__ import annotations


def build_policy_net(n_types: int = 3):
    """Construct the policy network. Imports torch lazily."""
    import torch.nn as nn

    class PolicyNet(nn.Module):
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

        def forward(self, x):
            z = self.head(self.features(x))
            return self.type_head(z), self.coord_head(z)

    return PolicyNet()
