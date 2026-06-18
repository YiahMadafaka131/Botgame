"""Reward primitives for `BotEnv`.

`PPOTrainer` calls `reward_fn(prev_frame, action, next_frame) -> float` after
each step. The right reward is game-specific (game score from OCR, life bar
position, a "you died" template match, etc.), but several signals are generic
enough to reuse:

  - `pixel_diff_reward`     — encourages screen change (something happened)
  - `region_brightness_reward` — proxy for "score area lit up"
  - `template_match_reward` — match a pre-captured template (game-over, hit, ...)
  - `compose_rewards`       — weighted average of multiple rewards

Compose them like:
    reward_fn = compose_rewards([
        (pixel_diff_reward(), 0.2),
        (region_brightness_reward(x=900, y=80, w=180, h=60), 0.8),
    ])
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from ..dataset.schema import Action

RewardFn = Callable[[np.ndarray, Action, np.ndarray], float]


def pixel_diff_reward(scale: float = 1.0 / 25.0, downsample: int = 4) -> RewardFn:
    """Reward proportional to mean per-pixel L1 change between frames.

    `scale` brings typical values into roughly [0, 1]. `downsample` averages
    over a stride to keep this cheap; defaults are sized for ~60 fps.
    """
    def _r(prev: np.ndarray, _action: Action, nxt: np.ndarray) -> float:
        if prev.shape != nxt.shape:
            return 0.0
        a = prev[::downsample, ::downsample].astype(np.int16)
        b = nxt[::downsample, ::downsample].astype(np.int16)
        diff = float(np.abs(a - b).mean())
        return float(min(1.0, diff * scale))

    return _r


def region_brightness_reward(
    x: int, y: int, w: int, h: int, baseline: float = 0.0
) -> RewardFn:
    """Reward = mean luminance in a screen region, normalized to [0, 1].

    Use this as a cheap stand-in when your game's "score lit up" looks like
    a brightening HUD element. `baseline` is subtracted (then clipped) so you
    can null out the resting brightness of the region.
    """
    def _r(_prev: np.ndarray, _action: Action, nxt: np.ndarray) -> float:
        h0, w0 = nxt.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w0, x + w), min(h0, y + h)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        crop = nxt[y0:y1, x0:x1]
        lum = float(crop.mean()) / 255.0
        return float(max(0.0, min(1.0, lum - baseline)))

    return _r


def template_match_reward(
    template: np.ndarray, threshold: float = 0.85, polarity: float = 1.0
) -> RewardFn:
    """Reward when a template appears in the frame (normalized cross-correlation).

    `polarity=+1` → present = good (e.g. "score-up" graphic).
    `polarity=-1` → present = bad (e.g. "game-over" graphic) — return -1.0
    when matched and 0.0 otherwise so PPO learns to avoid it.

    Pure-numpy NCC; OK for small templates (< 64×64). For real games use
    OpenCV's `matchTemplate` instead — this one is here so the API is testable
    without an optional dep.
    """
    if template.ndim != 3:
        raise ValueError("template must be (H, W, 3) RGB")
    th, tw, _ = template.shape
    t_flat = template.astype(np.float32).reshape(-1)
    t_norm = t_flat - t_flat.mean()
    t_denom = float(np.linalg.norm(t_norm)) or 1.0

    def _r(_prev: np.ndarray, _action: Action, nxt: np.ndarray) -> float:
        H, W = nxt.shape[:2]
        if H < th or W < tw:
            return 0.0
        best = 0.0
        stride = max(1, min(th, tw) // 4)
        for yy in range(0, H - th + 1, stride):
            for xx in range(0, W - tw + 1, stride):
                patch = nxt[yy:yy + th, xx:xx + tw].astype(np.float32).reshape(-1)
                p_norm = patch - patch.mean()
                p_denom = float(np.linalg.norm(p_norm)) or 1.0
                ncc = float((p_norm * t_norm).sum() / (p_denom * t_denom))
                if ncc > best:
                    best = ncc
        if best >= threshold:
            return float(polarity)
        return 0.0

    return _r


def compose_rewards(weighted: Iterable[tuple[RewardFn, float]]) -> RewardFn:
    """Linear combination of reward fns. Weights need not sum to 1."""
    items = list(weighted)
    if not items:
        raise ValueError("compose_rewards needs at least one term")

    def _r(prev: np.ndarray, action: Action, nxt: np.ndarray) -> float:
        return float(sum(w * fn(prev, action, nxt) for fn, w in items))

    return _r
