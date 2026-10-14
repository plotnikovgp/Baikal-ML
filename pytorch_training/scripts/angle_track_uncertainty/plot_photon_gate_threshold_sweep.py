#!/usr/bin/env python3
"""Plot the precommitted photon-time cut sweep on 2019 nue2 A and B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path: Path) -> dict:
    with path.open() as stream:
        result = json.load(stream)
    values = result["photon_residual_gates"]
    thresholds = np.asarray([row["max_abs_ns"] for row in values], dtype=float)
    if not np.array_equal(thresholds, [5, 10, 20, 30, 50, 100, 200]):
        raise ValueError(f"Unexpected time-gate sweep in {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    samples = {
        "2019-A (selection)": read(args.calibration),
        "2019-B (holdout)": read(args.holdout),
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), layout="constrained")
    colors = ("#2f6db0", "#c05a3c")
    for (label, sample), color in zip(samples.items(), colors):
        values = sample["photon_residual_gates"]
        x = np.asarray([row["max_abs_ns"] for row in values], dtype=float)
        for axis, q_index in zip(axes[:2], (0, 1)):
            y = [row["q50_q68_q95_deg"][q_index] for row in values]
            axis.plot(x, y, "o-", color=color, linewidth=2, markersize=4, label=label)
            axis.axhline(sample["baseline"][q_index], color=color, linestyle=":", linewidth=1)
        axes[2].plot(
            x,
            [100 * row["marginal_truth_signal_fraction"] for row in values],
            "o-",
            color=color,
            linewidth=2,
            markersize=4,
            label=label,
        )
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xticks([5, 10, 20, 50, 100, 200], labels=["5", "10", "20", "50", "100", "200"])
        axis.grid(alpha=0.23)
        axis.axvline(50, color="0.45", linestyle="--", linewidth=1, alpha=0.55)
        axis.set_xlabel("Photon residual limit [ns]")
    axes[0].set_ylabel("Angular error q50 [deg]")
    axes[1].set_ylabel("Angular error q68 [deg]")
    axes[2].set_ylabel("Added-hit truth-signal fraction [%]")
    axes[0].set_title("Median direction error")
    axes[1].set_title("68th-percentile direction error")
    axes[2].set_title("Added-hit purity on MC")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Frozen single-model input gate: 2019 nue2 A selection, B holdout", fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
