#!/usr/bin/env python3
"""Plot independent-test error and calibration diagnostics for the mixed model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def load_domain(saved: np.lib.npyio.NpzFile, prefix: str, factors: dict) -> dict:
    valid = saved[f"{prefix}_valid"].astype(bool)
    return {
        "angle": (
            saved[f"{prefix}_angle"],
            saved[f"{prefix}_sigma_angle"],
            factors["angle"],
        ),
        "line": (
            saved[f"{prefix}_line"][valid],
            saved[f"{prefix}_sigma_line"][valid],
            factors["line"],
        ),
    }


def quantile_groups(values: np.ndarray, count: int = 20):
    edges = np.unique(np.quantile(values, np.linspace(0, 1, count + 1)))
    index = np.searchsorted(edges[1:-1], values, side="right")
    return [index == i for i in range(len(edges) - 1)]


def plot_task(task: str, domains: dict, output_dir: Path) -> None:
    title, unit = (
        ("Angular error", "degrees")
        if task == "angle"
        else (
            "Transverse track-point error",
            "m",
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), layout="constrained")
    fig.suptitle(f"{title}: predicted versus observed 68% radius")
    for ax, (name, color) in zip(axes, (("pred", "#2463a5"), ("gt", "#c76a23"))):
        error, sigma, factors = domains[name][task]
        radius = sigma * factors["68"]
        limit = float(max(np.quantile(radius, 0.98), np.quantile(error, 0.98)))
        shown = (radius <= limit) & (error <= limit)
        ax.hexbin(
            radius[shown],
            error[shown],
            gridsize=65,
            extent=(0, limit, 0, limit),
            mincnt=1,
            cmap="Blues" if name == "pred" else "Oranges",
            norm=LogNorm(),
            linewidths=0,
        )
        groups = quantile_groups(radius, 24)
        centers = np.array([np.median(radius[group]) for group in groups if group.any()])
        q68 = np.array([np.quantile(error[group], 0.68) for group in groups if group.any()])
        visible = (centers <= limit) & (q68 <= limit)
        ax.plot([0, limit], [0, limit], "k--", lw=1.5, label="Ideal")
        ax.plot(
            centers[visible],
            q68[visible],
            color=color,
            lw=2,
            marker="o",
            ms=3,
            label="Observed q68 by bin",
        )
        coverage = np.mean(error <= radius)
        ax.text(
            0.03,
            0.97,
            f"N={len(error):,}; coverage={coverage:.1%}",
            va="top",
            transform=ax.transAxes,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        ax.set(
            title="Predicted signal hits" if name == "pred" else "GT signal hits (nue2)",
            xlabel=f"Predicted 68% radius [{unit}]",
            ylabel=f"Observed error [{unit}]",
            xlim=(0, limit),
            ylim=(0, limit),
        )
        ax.grid(alpha=0.2)
    axes[0].legend(loc="lower right")
    fig.savefig(output_dir / f"{task}_calibration.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.1), layout="constrained")
    fig.suptitle(f"{title}: coverage conditional on predicted difficulty")
    for ax, name in zip(axes, ("pred", "gt")):
        error, sigma, factors = domains[name][task]
        radius68 = sigma * factors["68"]
        radius95 = sigma * factors["95"]
        groups = quantile_groups(radius68, 16)
        centers = np.array([np.median(radius68[group]) for group in groups if group.any()])
        cov68 = [np.mean(error[group] <= radius68[group]) for group in groups if group.any()]
        cov95 = [np.mean(error[group] <= radius95[group]) for group in groups if group.any()]
        ax.axhline(0.68, color="#2463a5", ls="--", lw=1.3)
        ax.axhline(0.95, color="#bc3838", ls="--", lw=1.3)
        ax.plot(centers, cov68, color="#2463a5", marker="o", ms=3, label="68% radius")
        ax.plot(centers, cov95, color="#bc3838", marker="s", ms=3, label="95% radius")
        ax.set(
            title="Predicted signal hits" if name == "pred" else "GT signal hits (nue2)",
            xlabel=f"Predicted 68% radius [{unit}]",
            ylabel="Empirical coverage",
            xscale="log",
            ylim=(0.3, 1.02),
        )
        ax.grid(alpha=0.2)
    axes[0].legend(loc="lower left")
    fig.savefig(output_dir / f"{task}_conditional_coverage.png", dpi=180)
    plt.close(fig)


def plot_distributions(domains: dict, output_dir: Path) -> None:
    for task, unit in (("angle", "degrees"), ("line", "m")):
        pred = domains["pred"][task][0]
        gt = domains["gt"][task][0]
        max_error = max(np.quantile(pred, 0.95), np.quantile(gt, 0.95))
        bins = np.linspace(0, max_error, 100)
        fig, ax = plt.subplots(figsize=(9, 5.3), layout="constrained")
        for data, label, color in (
            (pred, f"Predicted hits (N={len(pred):,})", "#2463a5"),
            (gt, f"GT hits, nue2 (N={len(gt):,})", "#c76a23"),
        ):
            ax.hist(
                data,
                bins=bins,
                weights=np.ones(len(data)) / len(data),
                histtype="step",
                lw=2,
                color=color,
                label=label,
            )
            q50, q68 = np.quantile(data, [0.5, 0.68])
            ax.axvline(q50, color=color, lw=1.5)
            ax.axvline(q68, color=color, lw=1.5, ls="--")
        ax.set(
            title="Angular error distribution"
            if task == "angle"
            else "Transverse track-point error distribution",
            xlabel=f"Actual error [{unit}]",
            ylabel="Fraction of events per bin",
            xlim=(0, max_error),
        )
        ax.grid(alpha=0.2)
        ax.legend(title="Solid: q50; dashed: q68")
        fig.savefig(output_dir / f"{task}_error_distribution.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(args.metrics.read_text())
    with np.load(args.predictions) as saved:
        domains = {
            "pred": load_domain(saved, "pred", metrics["calibration_from_val"]),
            "gt": load_domain(saved, "gt", metrics["calibration_gt_from_val"]),
        }
    for task in ("angle", "line"):
        plot_task(task, domains, args.output_dir)
    plot_distributions(domains, args.output_dir)
    print(f"Saved 6 figures to {args.output_dir}")


if __name__ == "__main__":
    main()
