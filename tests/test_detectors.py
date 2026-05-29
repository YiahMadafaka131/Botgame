import json

import pytest

from botgame.redteam import (
    aggregate_by_level,
    composite_detector,
    coord_cluster_detector,
    default_composite,
    perfect_aim_detector,
    periodicity_detector,
    reaction_time_detector,
    run_sweep,
    SweepResult,
)


def _write(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


# ---- periodicity ----------------------------------------------------


def test_periodicity_robotic_high(tmp_path):
    path = tmp_path / "robotic.jsonl"
    _write(path, [{"ts": i * 0.5, "action": "tap", "target": [0, 0], "actual": [0, 0]}
                  for i in range(10)])
    assert periodicity_detector(str(path)) > 0.95


def test_periodicity_human_low(tmp_path):
    path = tmp_path / "jittery.jsonl"
    # Wildly varying intervals → CV high → score low.
    ts_offsets = [0.1, 0.9, 0.2, 1.4, 0.3, 1.7, 0.25, 0.95]
    t = 0.0
    recs = []
    for dt in ts_offsets:
        t += dt
        recs.append({"ts": t, "action": "tap", "target": [0, 0], "actual": [0, 0]})
    _write(path, recs)
    assert periodicity_detector(str(path)) < 0.3


def test_periodicity_too_few_actions_returns_zero(tmp_path):
    path = tmp_path / "short.jsonl"
    _write(path, [{"ts": 0.0, "action": "tap"}])
    assert periodicity_detector(str(path)) == 0.0


# ---- coord cluster --------------------------------------------------


def test_coord_cluster_robotic_high(tmp_path):
    path = tmp_path / "same_pixel.jsonl"
    _write(path, [{"ts": i * 0.5, "action": "tap",
                   "target": [500, 500], "actual": [500, 500]}
                  for i in range(10)])
    assert coord_cluster_detector(str(path)) == pytest.approx(1.0)


def test_coord_cluster_dispersed_low(tmp_path):
    path = tmp_path / "spread.jsonl"
    _write(path, [{"ts": i * 0.5, "action": "tap",
                   "target": [500, 500],
                   "actual": [500 + (i * 53) % 200, 500 + (i * 71) % 200]}
                  for i in range(20)])
    assert coord_cluster_detector(str(path)) < 0.3


# ---- perfect aim ----------------------------------------------------


def test_perfect_aim_all_perfect(tmp_path):
    path = tmp_path / "aimbot.jsonl"
    _write(path, [{"ts": 0.0, "action": "tap",
                   "target": [100, 200], "actual": [100, 200]}
                  for _ in range(5)])
    assert perfect_aim_detector(str(path)) == 1.0


def test_perfect_aim_with_jitter(tmp_path):
    path = tmp_path / "jittered.jsonl"
    _write(path, [{"ts": 0.0, "action": "tap",
                   "target": [100, 200], "actual": [101, 199]}
                  for _ in range(5)])
    assert perfect_aim_detector(str(path)) == 0.0


# ---- reaction time --------------------------------------------------


def test_reaction_time_zero_delay_high(tmp_path):
    path = tmp_path / "reflex.jsonl"
    _write(path, [{"ts": 0.0, "action": "tap", "reaction_s": 0.0,
                   "target": [0, 0], "actual": [0, 0]}
                  for _ in range(5)])
    assert reaction_time_detector(str(path)) == 1.0


def test_reaction_time_human_low(tmp_path):
    path = tmp_path / "human.jsonl"
    _write(path, [{"ts": 0.0, "action": "tap", "reaction_s": 0.25,
                   "target": [0, 0], "actual": [0, 0]}
                  for _ in range(5)])
    assert reaction_time_detector(str(path)) == 0.0


# ---- composite -----------------------------------------------------


def test_composite_averages_components(tmp_path):
    path = tmp_path / "mix.jsonl"
    # Perfectly periodic AND perfectly aimed AND zero reaction → all 1s.
    _write(path, [{"ts": i * 0.5, "action": "tap", "reaction_s": 0.0,
                   "target": [500, 500], "actual": [500, 500]}
                  for i in range(10)])
    assert default_composite(str(path)) == pytest.approx(1.0)


def test_composite_rejects_empty_list():
    with pytest.raises(ValueError):
        composite_detector([])


def test_composite_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        composite_detector([periodicity_detector], weights=[0.5, 0.5])


def test_composite_custom_weights(tmp_path):
    path = tmp_path / "mix.jsonl"
    _write(path, [{"ts": i * 0.5, "action": "tap", "reaction_s": 0.0,
                   "target": [500, 500], "actual": [500, 500]}
                  for i in range(10)])
    # Single detector with weight=1 reduces to the detector itself.
    det = composite_detector([periodicity_detector], weights=[1.0])
    assert det(str(path)) == pytest.approx(periodicity_detector(str(path)))


# ---- multi-session sweep -------------------------------------------


def test_run_sweep_with_multiple_sessions_per_level(tmp_path):
    runs: list[tuple[float, int, str]] = []

    def factory(level, seed, path):
        runs.append((level, seed, path))
        _write(path, [{"ts": 0.0, "action": "tap", "reaction_s": 0.0,
                       "target": [0, 0], "actual": [0, 0]}])
        return path

    def det(_):
        return 0.7

    results = run_sweep(
        factory, det, [0.0, 0.5],
        sessions_per_level=3, out_dir=str(tmp_path),
    )
    assert len(results) == 6
    seeds = [r.seed for r in results]
    assert len(set(seeds)) == 6, "each session must get a unique seed"


def test_aggregate_by_level_returns_mean_std_n():
    results = [
        SweepResult(level=0.0, seed=0, telemetry="", score=1.0),
        SweepResult(level=0.0, seed=1, telemetry="", score=0.8),
        SweepResult(level=1.0, seed=2, telemetry="", score=0.2),
    ]
    rows = aggregate_by_level(results)
    assert len(rows) == 2
    by_level = {r["level"]: r for r in rows}
    assert by_level[0.0]["mean"] == pytest.approx(0.9)
    assert by_level[0.0]["n"] == 2
    assert by_level[0.0]["std"] > 0
    assert by_level[1.0]["std"] == 0.0


def test_sessions_per_level_rejects_zero(tmp_path):
    def factory(level, seed, path):  # pragma: no cover
        return path
    with pytest.raises(ValueError):
        run_sweep(factory, lambda _: 0.0, [0.0],
                  sessions_per_level=0, out_dir=str(tmp_path))
