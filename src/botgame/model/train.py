"""Behavioral-cloning trainer (lazy torch import).

Trains the policy net to imitate the actions in a build-dataset directory:
cross-entropy on the action type plus coordinate regression (applied only to
real tap/swipe samples, since NOOP has no meaningful position).

Two details matter for a policy that "acts when it should":

  - **Class weighting.** Most frames in a recording are NOOP; unweighted
    cross-entropy collapses to "never act". Classes are weighted by inverse
    frequency so taps/swipes carry as much gradient as the idle majority.
  - **Temporal validation split.** The last `val_split` of the recording is
    held out (consecutive frames are near-duplicates, so a random split would
    leak). Per-class accuracy and coordinate error are reported so you can
    tell whether the model actually learned before putting it on a device.
"""

from __future__ import annotations

from collections import Counter

from .dataset import build_torch_dataset, read_labels
from .encoding import ACTION_TYPES, NOOP_INDEX
from .net import build_policy_net


def class_weights(type_counts: list[int]) -> list[float]:
    """Inverse-frequency weights, normalized to mean 1. Absent classes get 1."""
    total = sum(type_counts)
    if total == 0:
        return [1.0] * len(type_counts)
    raw = [total / c if c > 0 else 0.0 for c in type_counts]
    present = [w for w in raw if w > 0]
    mean = sum(present) / len(present)
    return [w / mean if w > 0 else 1.0 for w in raw]


def _evaluate(net, loader, screen_size, device):
    """Return (per-class accuracy dict, mean coord error in px) on a loader."""
    import torch

    width, height = screen_size
    correct = Counter()
    seen = Counter()
    px_err_sum, px_err_n = 0.0, 0

    net.eval()
    with torch.no_grad():
        for frames, types, coords in loader:
            frames, types, coords = frames.to(device), types.to(device), coords.to(device)
            type_logits, coord_pred = net(frames)
            pred = type_logits.argmax(dim=1)
            for cls in range(len(ACTION_TYPES)):
                m = types == cls
                seen[cls] += int(m.sum())
                correct[cls] += int((pred[m] == cls).sum())
            act = types != NOOP_INDEX
            if int(act.sum()) > 0:
                scale = torch.tensor(
                    [width, height, width, height], dtype=torch.float32, device=device
                )
                err = ((coord_pred[act] - coords[act]) * scale).abs().mean(dim=1)
                px_err_sum += float(err.sum())
                px_err_n += int(act.sum())
    net.train()

    acc = {
        ACTION_TYPES[c].value: (correct[c] / seen[c] if seen[c] else None)
        for c in range(len(ACTION_TYPES))
    }
    return acc, (px_err_sum / px_err_n if px_err_n else None)


def _format_metrics(acc: dict, px_err: float | None) -> str:
    parts = [
        f"{name}={v:.0%}" for name, v in acc.items() if v is not None
    ]
    if px_err is not None:
        parts.append(f"coord_err={px_err:.0f}px")
    return "  ".join(parts) if parts else "n/a"


def train(
    dataset_dir: str,
    screen_size: tuple[int, int],
    *,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    stack: int = 4,
    val_split: float = 0.1,
    out_path: str = "policy.pt",
    device: str | None = None,
) -> str:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = build_torch_dataset(dataset_dir, screen_size, stack=stack)
    if len(ds) == 0:
        raise ValueError(f"no samples found in {dataset_dir}")

    # Hold out the tail of the recording for validation.
    n_val = int(len(ds) * val_split)
    train_ds = Subset(ds, range(len(ds) - n_val)) if n_val else ds
    val_loader = (
        DataLoader(Subset(ds, range(len(ds) - n_val, len(ds))), batch_size=batch_size)
        if n_val
        else None
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    counts = Counter(rec["type"] for rec in read_labels(dataset_dir))
    type_counts = [counts.get(t.value, 0) for t in ACTION_TYPES]
    weights = torch.tensor(class_weights(type_counts), dtype=torch.float32, device=dev)
    print(
        "samples per class: "
        + "  ".join(f"{t.value}={c}" for t, c in zip(ACTION_TYPES, type_counts))
    )

    sample_frame, _, _ = ds[0]
    in_channels = sample_frame.shape[0]
    input_size = (sample_frame.shape[2], sample_frame.shape[1])  # (W, H)

    net = build_policy_net(n_types=len(ACTION_TYPES), in_channels=in_channels).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    net.train()
    for epoch in range(epochs):
        total = 0.0
        for frames, types, coords in loader:
            frames, types, coords = frames.to(dev), types.to(dev), coords.to(dev)
            type_logits, coord_pred = net(frames)
            type_loss = F.cross_entropy(type_logits, types, weight=weights)

            mask = (types != NOOP_INDEX).float().unsqueeze(1)
            coord_loss = (F.mse_loss(coord_pred, coords, reduction="none") * mask).sum()
            coord_loss = coord_loss / mask.sum().clamp(min=1.0)

            loss = type_loss + coord_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * frames.size(0)

        line = f"epoch {epoch + 1}/{epochs}  loss={total / len(train_ds):.4f}"
        if val_loader is not None:
            acc, px_err = _evaluate(net, val_loader, screen_size, dev)
            line += f"  val: {_format_metrics(acc, px_err)}"
        print(line)

    torch.save(
        {
            "state_dict": net.state_dict(),
            "screen_size": list(screen_size),
            "stack": stack,
            "in_channels": in_channels,
            "input_size": list(input_size),
        },
        out_path,
    )
    print(f"Saved policy to {out_path}")
    return out_path
