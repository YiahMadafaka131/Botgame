"""Behavior humanization for red-teaming your own bot detector.

The point is NOT a single "undetectable" bot. It is a tunable knob, `level`,
that sweeps behavior from fully robotic (0.0) to human-like (1.0). Run your
games and detector across the sweep and you get a detection-rate curve that
tells you exactly which signals your detector relies on and where its gaps are.

Signals tuned here:
  - touch coordinate dispersion (robotic = same pixel every time)
  - reaction-time latency + jitter (robotic = 0 ms, acts on the frame)
  - inter-action interval + jitter (robotic = perfectly periodic)
  - swipe trajectory curvature + micro-tremor (robotic = straight line)

Everything is driven by a seedable RNG so experiments are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Point = tuple[int, int]


@dataclass
class Humanizer:
    """Tunable human-likeness. `level` in [0, 1]; 0 = robotic, 1 = human-like."""

    level: float = 0.0
    seed: int | None = None
    # Tuning constants (the values reached at level == 1.0).
    max_coord_sigma_px: float = 12.0
    base_reaction_s: float = 0.18
    reaction_jitter_s: float = 0.08
    interval_jitter_frac: float = 0.35
    max_curve_frac: float = 0.18
    tremor_px: float = 2.5
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.level <= 1.0:
            raise ValueError(f"level must be in [0, 1], got {self.level}")
        self._rng = np.random.default_rng(self.seed)

    # --- coordinates -----------------------------------------------------
    def jitter_point(self, x: int, y: int) -> Point:
        """Disperse a target pixel by a level-scaled Gaussian."""
        sigma = self.max_coord_sigma_px * self.level
        if sigma <= 0:
            return int(x), int(y)
        dx, dy = self._rng.normal(0.0, sigma, size=2)
        return int(round(x + dx)), int(round(y + dy))

    # --- timing ----------------------------------------------------------
    def reaction_delay(self) -> float:
        """Seconds to wait before acting (0 when robotic)."""
        if self.level <= 0:
            return 0.0
        base = self.base_reaction_s * self.level
        jitter = abs(self._rng.normal(0.0, self.reaction_jitter_s * self.level))
        return base + jitter

    def action_interval(self, base_interval_s: float) -> float:
        """Perturb a nominal interval between actions, keeping it positive."""
        if self.level <= 0:
            return base_interval_s
        jitter = self._rng.normal(0.0, base_interval_s * self.interval_jitter_frac * self.level)
        return max(0.0, base_interval_s + jitter)

    # --- trajectories ----------------------------------------------------
    def bezier_path(self, start: Point, end: Point, n_points: int = 16) -> list[Point]:
        """Quadratic Bezier from start to end, bowed and trembling by `level`.

        At level 0 this collapses to a straight, evenly sampled line.
        """
        x0, y0 = start
        x2, y2 = end
        n_points = max(2, n_points)

        length = float(np.hypot(x2 - x0, y2 - y0))
        # Control point at the midpoint, pushed perpendicular to the line.
        mx, my = (x0 + x2) / 2.0, (y0 + y2) / 2.0
        if length > 0 and self.level > 0:
            nx, ny = -(y2 - y0) / length, (x2 - x0) / length
            offset = self._rng.normal(0.0, self.max_curve_frac * length * self.level)
            cx, cy = mx + nx * offset, my + ny * offset
        else:
            cx, cy = mx, my

        ts = np.linspace(0.0, 1.0, n_points)
        path: list[Point] = []
        for t in ts:
            bx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t**2 * x2
            by = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t**2 * y2
            if self.level > 0:
                tx, ty = self._rng.normal(0.0, self.tremor_px * self.level, size=2)
                bx, by = bx + tx, by + ty
            path.append((int(round(bx)), int(round(by))))
        # Pin exact endpoints so the gesture lands where intended.
        path[0], path[-1] = (int(x0), int(y0)), (int(x2), int(y2))
        return path
