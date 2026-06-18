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

Each path-based detector has a row-based companion (`*_score`) that takes a
list of telemetry-shaped dicts and returns the same score. The row variants
are what `botgame.rl.detection_evasion_reward` uses to score the policy's
live action stream during PPO.
"""

from __future__ import annotations

import json
import math
from typing import Callable, Iterable

Detector = Callable[[str], float]
RowScorer = Callable[[list[dict]], float]


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


def _normalize_weights(weights: Iterable[float] | None, n: int) -> list[float]:
    if weights is None:
        return [1.0 / n] * n
    w = list(weights)
    if len(w) != n:
        raise ValueError("weights length must match scorers length")
    total = sum(w)
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    return [x / total for x in w]


def _var(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    return sum((x - mean) ** 2 for x in xs) / len(xs)


# ---- row-based scorers (the actual logic) ----------------------------


def periodicity_score(rows: list[dict]) -> float:
    """Bots tick on a metronome. Score = 1 - coefficient_of_variation(intervals).

    Robotic = constant interval = CV ≈ 0 = score ≈ 1.
    Human = jittered intervals = CV high = score ≈ 0.
    """
    actions = _action_records(rows)
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
    return _clip01(1.0 - cv / 0.5)


def coord_cluster_score(rows: list[dict]) -> float:
    """Bots land on the same pixel every time. Score = 1 - normalized_spread.

    Uses the std-dev of `actual` coords across taps, normalized by a typical
    "human" sigma (24 px). Below that → bot-like; above → human-like.
    """
    actions = [r for r in _action_records(rows)
               if r.get("action") == "tap" and "actual" in r]
    if len(actions) < 3:
        return 0.0
    xs = [r["actual"][0] for r in actions]
    ys = [r["actual"][1] for r in actions]
    sigma = math.sqrt(_var(xs) + _var(ys))
    return _clip01(1.0 - sigma / 24.0)


def perfect_aim_score(rows: list[dict]) -> float:
    """Fraction of taps where `actual == target` (no humanizer jitter applied)."""
    taps = [r for r in _action_records(rows)
            if r.get("action") == "tap" and "target" in r and "actual" in r]
    if not taps:
        return 0.0
    perfect = sum(1 for r in taps if r["target"] == r["actual"])
    return _clip01(perfect / len(taps))


def reaction_time_score(rows: list[dict]) -> float:
    """Bot reaction delays are 0 ms or implausibly low.

    Score = fraction of records with `reaction_s < 0.08` (sub-human reflex).
    Default cutoff 80 ms — below the canonical ~150 ms human reaction floor.
    """
    actions = _action_records(rows)
    if not actions:
        return 0.0
    subhuman = sum(1 for r in actions if r.get("reaction_s", 0.0) < 0.08)
    return _clip01(subhuman / len(actions))


def composite_score(
    scorers: Iterable[RowScorer],
    weights: Iterable[float] | None = None,
) -> RowScorer:
    """Combine several row scorers with a (normalized) weighted average."""
    scorer_list = list(scorers)
    if not scorer_list:
        raise ValueError("composite_score needs at least one scorer")
    w = _normalize_weights(weights, len(scorer_list))

    def _composite(rows: list[dict]) -> float:
        return _clip01(sum(weight * s(rows) for s, weight in zip(scorer_list, w)))

    return _composite


# ---- path-based detectors (wrap the row scorers) ---------------------


def periodicity_detector(telemetry_path: str) -> float:
    return periodicity_score(_load(telemetry_path))


def coord_cluster_detector(telemetry_path: str) -> float:
    return coord_cluster_score(_load(telemetry_path))


def perfect_aim_detector(telemetry_path: str) -> float:
    return perfect_aim_score(_load(telemetry_path))


def reaction_time_detector(telemetry_path: str) -> float:
    return reaction_time_score(_load(telemetry_path))


def composite_detector(
    detectors: Iterable[Detector],
    weights: Iterable[float] | None = None,
) -> Detector:
    """Combine several detectors with a (normalized) weighted average."""
    det_list = list(detectors)
    if not det_list:
        raise ValueError("composite_detector needs at least one detector")
    w = _normalize_weights(weights, len(det_list))

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


default_composite_score: RowScorer = composite_score(
    [
        periodicity_score,
        coord_cluster_score,
        perfect_aim_score,
        reaction_time_score,
    ],
)
"""Row-scorer twin of `default_composite` — usable as the RL evasion scorer."""


