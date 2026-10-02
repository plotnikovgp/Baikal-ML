#!/usr/bin/env python3
"""Plot transverse track-distance error distributions from saved test predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "count": int(len(values)),
        "q50_m": float(np.quantile(values, 0.50)),
        "q68_m": float(np.quantile(values, 0.68)),
        "q90_m": float(np.quantile(values, 0.90)),
        "q95_m": float(np.quantile(values, 0.95)),
        "q99_m": float(np.quantile(values, 0.99)),
    }


def load_tasks(path: Path):
    with np.load(path) as pred:
        valid = pred["valid"].astype(bool)
        sigma_line = pred["sigma_line"][valid].astype(np.float64)
        sigma_angle = pred["sigma_angle"][valid].astype(np.float64)
        sigma_200 = np.sqrt(sigma_line**2 + (200.0 * np.deg2rad(sigma_angle)) ** 2)
        return {
            "reference": {
                "title": "At track reference point",
                "definition": "Transverse anchor error relative to true direction",
                "error": pred["line"][valid].astype(np.float64),
                "scale": sigma_line,
            },
            "plus_200_m": {
                "title": "At +200 m along true track",
                "definition": "Distance from true track point to predicted line",
                "error": pred["line200"][valid].astype(np.float64),
                "scale": sigma_200,
            },
        }


def plot_distributions(tasks: dict, output: Path):
    columns = len(tasks)
    fig, axes = plt.subplots(
        2, columns, figsize=(7.5 * columns, 9.2), layout="constrained", squeeze=False
    )
    fig.suptitle("Transverse track-anchor error - independent nue2 test", fontsize=16)
    for column, task in enumerate(tasks.values()):
        error = task["error"]
        q50, q68, q95 = np.quantile(error, [0.50, 0.68, 0.95])
        core_limit = 80.0
        core = error <= core_limit
        ax = axes[0, column]
        ax.hist(
            error[core],
            bins=np.linspace(0, core_limit, 81),
            weights=np.full(int(core.sum()), 100.0 / len(error)),
            color="#357cb8",
            alpha=0.88,
        )
        ax.axvline(q50, color="#d9483b", lw=2, label=f"q50 = {q50:.1f} m")
        ax.axvline(q68, color="#a02f6b", lw=2, label=f"q68 = {q68:.1f} m")
        ax.set(
            title=f"{task['title']} - core ({core.mean():.1%} shown)",
            xlabel="Distance error [m]",
            ylabel="Events per 1-m bin [%]",
            xlim=(0, core_limit),
        )
        ax.grid(alpha=0.22)
        ax.legend(loc="upper right")

        ax = axes[1, column]
        tail_bins = np.geomspace(0.05, 650.0, 90)
        ax.hist(
            error,
            bins=tail_bins,
            weights=np.full(len(error), 100.0 / len(error)),
            color="#357cb8",
            alpha=0.88,
        )
        for value, label, color in (
            (q50, "q50", "#d9483b"),
            (q68, "q68", "#a02f6b"),
            (q95, "q95", "#c47a16"),
        ):
            ax.axvline(value, color=color, lw=2, label=f"{label} = {value:.1f} m")
        ax.set(
            title=f"{task['title']} - full tail",
            xlabel="Distance error [m]",
            ylabel="Events per log-spaced bin [%]",
            xscale="log",
            yscale="log",
            xlim=(0.05, 650.0),
        )
        ax.grid(alpha=0.22, which="both")
        ax.legend(loc="upper left")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_quartile_cdfs(tasks: dict, output: Path):
    columns = len(tasks)
    fig, axes = plt.subplots(
        1, columns, figsize=(7.5 * columns, 5.8), layout="constrained", squeeze=False
    )
    fig.suptitle("Track-anchor error by predicted uncertainty quartile", fontsize=16)
    colors = ["#2479a8", "#36a37c", "#d39a2d", "#c54648"]
    for ax, task in zip(axes[0], tasks.values()):
        error = task["error"]
        scale = task["scale"]
        edges = np.quantile(scale, [0, 0.25, 0.5, 0.75, 1.0])
        group_index = np.searchsorted(edges[1:-1], scale, side="right")
        grid = np.geomspace(0.05, 650.0, 900)
        for quartile in range(4):
            values = np.sort(error[group_index == quartile])
            cdf = np.searchsorted(values, grid, side="right") / len(values)
            q68 = np.quantile(values, 0.68)
            ax.plot(
                grid,
                cdf,
                color=colors[quartile],
                lw=2.2,
                label=f"Q{quartile + 1}: q68 = {q68:.1f} m",
            )
        ax.axhline(0.68, color="black", ls="--", lw=1.2)
        ax.set(
            title=task["title"],
            xlabel="Distance error [m]",
            ylabel="Cumulative event fraction",
            xscale="log",
            xlim=(0.05, 650.0),
            ylim=(0, 1.01),
        )
        ax.grid(alpha=0.22, which="both")
        ax.legend(loc="lower right")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--include-plus200", action="store_true", help="Include optional 200-m lever-arm diagnostic"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.predictions)
    if not args.include_plus200:
        tasks = {"reference": tasks["reference"]}
    plot_distributions(tasks, args.output_dir / "reference_distance_error_distribution.png")
    plot_quartile_cdfs(tasks, args.output_dir / "reference_distance_error_by_uncertainty.png")
    metrics = {
        key: {
            "definition": task["definition"],
            "overall": summarize(task["error"]),
            "quartiles": [
                summarize(
                    task["error"][
                        np.searchsorted(
                            np.quantile(task["scale"], [0.25, 0.5, 0.75]),
                            task["scale"],
                            side="right",
                        )
                        == index
                    ]
                )
                for index in range(4)
            ],
        }
        for key, task in tasks.items()
    }
    (args.output_dir / "track_distance_error_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
