"""Plot the detector-vs-level curve from a sweep CSV.

matplotlib is an optional dependency, lazy-imported so the rest of the package
keeps working without it. The plot shows mean detection score per level with
error bars (one std dev across sessions at that level), so you can see both
the curve and how reproducible it is.
"""

from __future__ import annotations

import csv
from collections import defaultdict


def plot_sweep(csv_path: str, out_path: str, title: str | None = None) -> str:
    """Read a sweep CSV and write a PNG showing detection score vs level.

    Returns `out_path` for chaining.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless-safe
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plot_sweep. Install with `pip install matplotlib`."
        ) from exc

    buckets: dict[float, list[float]] = defaultdict(list)
    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            buckets[float(row["level"])].append(float(row["score"]))

    levels = sorted(buckets)
    means = [sum(buckets[l]) / len(buckets[l]) for l in levels]
    stds = []
    for l in levels:
        scores = buckets[l]
        if len(scores) > 1:
            mean = sum(scores) / len(scores)
            stds.append((sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5)
        else:
            stds.append(0.0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(levels, means, yerr=stds, marker="o", capsize=4, linewidth=2)
    ax.set_xlabel("humanization level")
    ax.set_ylabel("detector score (P[bot])")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(min(levels) - 0.05, max(levels) + 0.05)
    ax.grid(True, alpha=0.3)
    ax.set_title(title or "Detector score vs humanization level")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
