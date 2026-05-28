"""Behavioral-cloning trainer (lazy torch import).

Trains the policy net to imitate the actions in a build-dataset directory:
cross-entropy on the action type plus coordinate regression (applied only to
real tap/swipe samples, since NOOP has no meaningful position).
"""

from __future__ import annotations

from .dataset import build_torch_dataset
from .encoding import ACTION_TYPES
from .net import build_policy_net


def train(
    dataset_dir: str,
    screen_size: tuple[int, int],
    *,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    out_path: str = "policy.pt",
    device: str | None = None,
) -> str:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = build_torch_dataset(dataset_dir, screen_size)
    if len(ds) == 0:
        raise ValueError(f"no samples found in {dataset_dir}")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    net = build_policy_net(n_types=len(ACTION_TYPES)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    noop_index = 0  # ACTION_TYPES[0] is NOOP

    net.train()
    for epoch in range(epochs):
        total = 0.0
        for frames, types, coords in loader:
            frames, types, coords = frames.to(dev), types.to(dev), coords.to(dev)
            type_logits, coord_pred = net(frames)
            type_loss = F.cross_entropy(type_logits, types)

            mask = (types != noop_index).float().unsqueeze(1)
            coord_loss = (F.mse_loss(coord_pred, coords, reduction="none") * mask).sum()
            coord_loss = coord_loss / mask.sum().clamp(min=1.0)

            loss = type_loss + coord_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * frames.size(0)
        print(f"epoch {epoch + 1}/{epochs}  loss={total / len(ds):.4f}")

    torch.save({"state_dict": net.state_dict(), "screen_size": list(screen_size)}, out_path)
    print(f"Saved policy to {out_path}")
    return out_path
