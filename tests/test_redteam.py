import csv
import json
import os

import pytest

from botgame.redteam import SweepResult, run_sweep, sample_detector, load_detector


def _write_session(path: str, level: float, steps: int = 5) -> None:
    """Write JSONL telemetry mimicking `humanize.Humanizer(level=level)`."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(steps):
            target = [500, 500]
            if level == 0.0:
                actual = list(target)
                react = 0.0
            else:
                actual = [500 + i, 500 - i]
                react = 0.05 + 0.01 * i
            fh.write(json.dumps({
                "ts": 0.0, "action": "tap",
                "target": target, "actual": actual,
                "reaction_s": react, "level": level,
            }) + "\n")


def test_run_sweep_runs_one_session_per_level_and_writes_csv(tmp_path):
    calls: list[tuple[float, int, str]] = []

    def factory(level, seed, path):
        calls.append((level, seed, path))
        _write_session(path, level=level)
        return path

    levels = [0.0, 0.5, 1.0]
    out_dir = tmp_path / "rt"
    results = run_sweep(factory, sample_detector, levels, out_dir=str(out_dir))

    assert [r.level for r in results] == levels
    assert all(isinstance(r, SweepResult) for r in results)
    assert len(calls) == 3
    # Score must decrease as level rises (more variance → less "bot-looking").
    assert results[0].score > results[-1].score

    csv_path = out_dir / "results.csv"
    assert csv_path.exists()
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert rows[0]["level"] == "0.0"


def test_sample_detector_pegs_robotic_session_high(tmp_path):
    path = tmp_path / "robotic.jsonl"
    _write_session(str(path), level=0.0, steps=10)
    score = sample_detector(str(path))
    assert score == 1.0


def test_sample_detector_lowers_score_with_variance(tmp_path):
    path = tmp_path / "human.jsonl"
    _write_session(str(path), level=1.0, steps=10)
    score = sample_detector(str(path))
    assert score < 0.5


def test_sample_detector_ignores_noops(tmp_path):
    path = tmp_path / "noops.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for _ in range(5):
            fh.write(json.dumps({"ts": 0.0, "action": "noop", "level": 0.0}) + "\n")
    assert sample_detector(str(path)) == 0.0


def test_run_sweep_rejects_out_of_range_levels(tmp_path):
    def factory(level, seed, path):  # pragma: no cover - not reached
        return path
    with pytest.raises(ValueError):
        run_sweep(factory, sample_detector, [0.0, 1.5], out_dir=str(tmp_path))


def test_run_sweep_rejects_detector_out_of_range(tmp_path):
    def factory(level, seed, path):
        _write_session(path, level)
        return path

    def bad_detector(_):
        return 2.0  # out of [0, 1]

    with pytest.raises(ValueError, match="probability"):
        run_sweep(factory, bad_detector, [0.0], out_dir=str(tmp_path))


def test_load_detector_resolves_import_spec():
    fn = load_detector("botgame.redteam.sweep:sample_detector")
    assert fn is sample_detector


def test_load_detector_rejects_bad_spec():
    with pytest.raises(ValueError):
        load_detector("no_colon_here")
