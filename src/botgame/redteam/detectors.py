"""Detector primitives for the red-team sweep.

Each detector reads a telemetry JSONL file and returns a probability in
[0, 1] that the session was produced by a bot (1.0 = "obviously a bot",
0.0 = "looks human"). They're intentionally simple and feature-explicit so
you can read the score and know *which* signal triggered.

Plug them into the sweep via `botgame redteam --detector <import-spec>`:

    botgame.redteam.detectors:periodicity_detector
    botgame.redteam.detectors:coord_cluster_detector
    botgame.redteam.detectors:perfect_aim_detector
    botgame.redteam.detectors:reaction_time_detector
    botgame.redteam.detectors:default_composite

Or build your own composite:

    from botgame.redteam.detectors import composite_detector, periodicity_detector
    my_det = composite_detector([periodicity_detector, perfect_aim_detector],
                                weights=[0.7, 0.3])
"""

from __future__ import annotations

import json
import math
from typing import Callable, Iterable

Detector = Callable[[str], float]


# ---- shared loaders --------------------------------------------------


def _load(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _action_records(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("action") in ("tap", "swipe")]


def _clip01(x: float) -> float:
    if math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, x))


# ---- individual detectors --------------------------------------------


def periodicity_detector(telemetry_path: str) -> float:
    """Bots tick on a metronome. Score = 1 - coefficient_of_variation(intervals).

    Robotic = constant interval = CV ≈ 0 = score ≈ 1.
    Human = jittered intervals = CV high = score ≈ 0.
    """
    actions = _action_records(_load(telemetry_path))
    if len(actions) < 3:
        return 0.0
    ts = [r["ts"] for r in actions]
    intervals = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    intervals = [x for x in intervals if x > 0]
    if len(intervals) < 2:
        return 0.0
    mean = sum(intervals) / len(intervals)
    if mean <= 0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in intervals) / len(intervals)
    cv = math.sqrt(variance) / mean
    # CV ~ 0 (robotic) → 1; CV >= 0.5 (jittery) → 0.
    return _clip01(1.0 - cv / 0.5)


def coord_cluster_detector(telemetry_path: str) -> float:
    """Bots land on the same pixel every time. Score = 1 - normalized_spread.

    Uses the std-dev of `actual` coords across taps, normalized by a typical
    "human" sigma (24 px). Below that → bot-like; above → human-like.
    """
    actions = [r for r in _action_records(_load(telemetry_path))
               if r.get("action") == "tap" and "actual" in r]
    if len(actions) < 3:
        return 0.0
    xs = [r["actual"][0] for r in actions]
    ys = [r["actual"][1] for r in actions]
    sigma = math.sqrt(_var(xs) + _var(ys))
    return _clip01(1.0 - sigma / 24.0)


def perfect_aim_detector(telemetry_path: str) -> float:
    """Fraction of taps where `actual == target` (no humanizer jitter applied)."""
    taps = [r for r in _action_records(_load(telemetry_path))
            if r.get("action") == "tap" and "target" in r and "actual" in r]
    if not taps:
        return 0.0
    perfect = sum(1 for r in taps if r["target"] == r["actual"])
    return _clip01(perfect / len(taps))


def reaction_time_detector(telemetry_path: str) -> float:
    """Bot reaction delays are 0 ms or implausibly low.

    Score = fraction of records with `reaction_s < 0.08` (sub-human reflex).
    Default cutoff 80 ms — below the canonical ~150 ms human reaction floor.
    """
    actions = _action_records(_load(telemetry_path))
    if not actions:
        return 0.0
    subhuman = sum(1 for r in actions if r.get("reaction_s", 0.0) < 0.08)
    return _clip01(subhuman / len(actions))


def composite_detector(
    detectors: Iterable[Detector],
    weights: Iterable[float] | None = None,
) -> Detector:
    """Combine several detectors with a (normalized) weighted average."""
    det_list = list(detectors)
    if not det_list:
        raise ValueError("composite_detector needs at least one detector")
    if weights is None:
        w = [1.0 / len(det_list)] * len(det_list)
    else:
        w = list(weights)
        if len(w) != len(det_list):
            raise ValueError("weights length must match detectors length")
        total = sum(w)
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        w = [x / total for x in w]

    def _composite(path: str) -> float:
        return _clip01(sum(weight * det(path) for det, weight in zip(det_list, w)))

    return _composite


default_composite: Detector = composite_detector(
    [
        periodicity_detector,
        coord_cluster_detector,
        perfect_aim_detector,
        reaction_time_detector,
    ],
)
"""Equal-weight composite of the four built-in detectors — good default."""


# ---- helpers ---------------------------------------------------------


def _var(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    return sum((x - mean) ** 2 for x in xs) / len(xs)
