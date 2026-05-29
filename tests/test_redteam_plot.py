"""Plot helper smoke tests. Skipped if matplotlib is missing."""

import csv

import pytest

pytest.importorskip("matplotlib")

from botgame.redteam.plot import plot_sweep


def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["level", "seed", "telemetry", "score"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_plot_sweep_writes_png(tmp_path):
    csv_path = tmp_path / "results.csv"
    _write_csv(csv_path, [
        {"level": 0.0, "seed": 0, "telemetry": "x", "score": 0.95},
        {"level": 0.0, "seed": 1, "telemetry": "x", "score": 0.92},
        {"level": 0.5, "seed": 2, "telemetry": "x", "score": 0.40},
        {"level": 1.0, "seed": 3, "telemetry": "x", "score": 0.05},
    ])
    out_path = tmp_path / "plot.png"
    returned = plot_sweep(str(csv_path), str(out_path))
    assert returned == str(out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_sweep_handles_single_session_per_level(tmp_path):
    csv_path = tmp_path / "results.csv"
    _write_csv(csv_path, [
        {"level": 0.0, "seed": 0, "telemetry": "x", "score": 1.0},
        {"level": 1.0, "seed": 1, "telemetry": "x", "score": 0.0},
    ])
    out_path = tmp_path / "plot.png"
    plot_sweep(str(csv_path), str(out_path))
    assert out_path.exists()
