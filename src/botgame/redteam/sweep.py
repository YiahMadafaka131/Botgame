"""Automated red-team sweep: run the bot across humanization levels and ask
your detector to score each session.

Inputs:
  - a `bot_factory(level, seed, telemetry_path)` that builds + runs the bot
    for one session and returns the path of the JSONL telemetry it produced.
  - a `detector(telemetry_path)` callable returning a confidence in [0, 1]
    (1 = "definitely a bot"). Plug in your own model here.
  - the list of levels to sweep (default 0.0, 0.25, 0.5, 0.75, 1.0).

Output: a list of `SweepResult(level, seed, telemetry, score)` plus a CSV
written to disk. Plot `score` vs `level` to see which signals your detector
relies on and where its gaps are.

The bot factory is fully decoupled so tests can drive the sweep with a fake
bot — no device required.
"""

from __future__ import annotations

import csv
import importlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Sequence

BotFactory = Callable[[float, int, str], str]
Detector = Callable[[str], float]


@dataclass
class SweepResult:
    level: float
    seed: int
    telemetry: str
    score: float


def run_sweep(
    bot_factory: BotFactory,
    detector: Detector,
    levels: Sequence[float],
    *,
    seed_base: int = 0,
    sessions_per_level: int = 1,
    out_dir: str = "redteam",
    csv_path: str | None = None,
) -> list[SweepResult]:
    """Run `sessions_per_level` sessions at each level, score each, write CSV.

    `bot_factory(level, seed, telemetry_path)` must run the bot to completion
    and write its telemetry to `telemetry_path` (returned as confirmation).

    Different seeds across sessions at the same level give detection-rate
    estimates with variance bars; one session per level is fine for smoke
    tests but rarely enough to draw conclusions.
    """
    for level in levels:
        if not 0.0 <= level <= 1.0:
            raise ValueError(f"level must be in [0, 1]; got {level}")
    if sessions_per_level < 1:
        raise ValueError("sessions_per_level must be >= 1")

    os.makedirs(out_dir, exist_ok=True)
    csv_path = csv_path or os.path.join(out_dir, "results.csv")
    results: list[SweepResult] = []

    counter = 0
    for level in levels:
        for _ in range(sessions_per_level):
            seed = seed_base + counter
            counter += 1
            telemetry = os.path.join(out_dir, f"level_{level:.2f}_seed_{seed}.jsonl")
            bot_factory(level, seed, telemetry)
            score = float(detector(telemetry))
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"detector returned {score}; expected a probability in [0, 1]"
                )
            results.append(SweepResult(level=level, seed=seed, telemetry=telemetry, score=score))

    _write_csv(csv_path, results)
    return results


def aggregate_by_level(results: Iterable[SweepResult]) -> list[dict]:
    """Collapse per-session results into (level, mean_score, std, n) rows."""
    buckets: dict[float, list[float]] = {}
    for r in results:
        buckets.setdefault(r.level, []).append(r.score)
    rows = []
    for level in sorted(buckets):
        scores = buckets[level]
        mean = sum(scores) / len(scores)
        if len(scores) > 1:
            std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
        else:
            std = 0.0
        rows.append({"level": level, "mean": mean, "std": std, "n": len(scores)})
    return rows


def _write_csv(path: str, results: Iterable[SweepResult]) -> None:
    rows = [asdict(r) for r in results]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["level", "seed", "telemetry", "score"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sample_detector(telemetry_path: str) -> float:
    """A demo detector — *not* production-grade.

    Heuristic: if every reaction delay and tap-jitter is zero, the actor is
    almost certainly a bot. The more variance, the lower the score. Drop in
    your own detector for real experiments.
    """
    n = 0
    zero_reaction = 0
    zero_jitter = 0
    with open(telemetry_path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["action"] == "noop":
                continue
            n += 1
            if rec.get("reaction_s", 0.0) == 0.0:
                zero_reaction += 1
            tgt = rec.get("target")
            act = rec.get("actual")
            if tgt is not None and act is not None and tgt == act:
                zero_jitter += 1
    if n == 0:
        return 0.0
    return 0.5 * (zero_reaction / n) + 0.5 * (zero_jitter / n)


def load_detector(spec: str) -> Detector:
    """Resolve a `module:function` import spec to a detector callable."""
    if ":" not in spec:
        raise ValueError(f"detector spec must be 'module:function', got {spec!r}")
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name)
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn
