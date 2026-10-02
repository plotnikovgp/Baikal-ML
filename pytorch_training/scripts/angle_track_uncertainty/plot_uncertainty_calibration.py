#!/usr/bin/env python3
"""Test-set calibration diagnostics for frozen nue2 direction/track uncertainty.

Run from the track_anchor source directory, next to train_track_uncertainty_v2.py.
Unlike signed theta/phi residuals in the thesis, these plots use nonnegative
geodesic and transverse-track errors, which match our radial uncertainty head.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from train_angle_signal import make_loader
from train_track_uncertainty_v2 import TrackUncertainty, load_base, make_dataset, predict


def get_tasks(pred: dict, calibration: dict) -> dict:
    valid = pred["valid"]
    sigma_200 = np.sqrt(
        pred["sigma_line"][valid] ** 2 + (200.0 * np.deg2rad(pred["sigma_angle"][valid])) ** 2
    )
    return {
        "angle": {
            "title": "Direction",
            "unit": "deg",
            "error": pred["angle"],
            "sigma": pred["sigma_angle"],
            "factors": calibration["angle"],
        },
        "line": {
            "title": "Track at reference point",
            "unit": "m",
            "error": pred["line"][valid],
            "sigma": pred["sigma_line"][valid],
            "factors": calibration["line"],
        },
        "line200": {
            "title": "Track at +200 m",
            "unit": "m",
            "error": pred["line200"][valid],
            "sigma": sigma_200,
            "factors": calibration["line200"],
        },
    }


def quantile_bins(x: np.ndarray, n_bins: int = 20):
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    index = np.searchsorted(edges[1:-1], x, side="right")
    return [(index == i) for i in range(len(edges) - 1)]


def plot_density(tasks: dict, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.7), layout="constrained")
    fig.suptitle(
        "Predicted 68% radius versus actual reconstruction error (independent nue2 test)",
        fontsize=15,
    )
    for ax, task in zip(axes, tasks.values()):
        error = task["error"]
        radius = task["sigma"] * task["factors"]["68"]
        limit = max(np.quantile(radius, 0.98), np.quantile(error, 0.98))
        shown = (radius <= limit) & (error <= limit)
        hb = ax.hexbin(
            radius[shown],
            error[shown],
            gridsize=66,
            extent=(0, limit, 0, limit),
            mincnt=1,
            cmap="Blues",
            norm=LogNorm(vmin=1, vmax=3000),
            linewidths=0,
        )
        bins = quantile_bins(radius, 24)
        center = np.array([np.median(radius[b]) for b in bins if b.any()])
        q68 = np.array([np.quantile(error[b], 0.68) for b in bins if b.any()])
        visible = (center <= limit) & (q68 <= limit)
        ax.plot([0, limit], [0, limit], color="black", lw=1.7, ls="--", label="Ideal 68%")
        ax.plot(
            center[visible],
            q68[visible],
            color="#d82e35",
            lw=2.4,
            marker="o",
            ms=3.4,
            label="Observed q68 by bin",
        )
        ax.set(
            xlim=(0, limit),
            ylim=(0, limit),
            title=task["title"],
            xlabel=f"Predicted 68% radius [{task['unit']}]",
            ylabel=f"Actual error [{task['unit']}]",
        )
        ax.grid(alpha=0.22)
        ax.text(
            0.03,
            0.96,
            f"N={len(error):,}; shown={shown.mean():.1%}",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        )
    axes[0].legend(loc="lower right", frameon=True)
    colorbar = fig.colorbar(hb, ax=axes, shrink=0.81, pad=0.01)
    colorbar.set_label("Events per hexbin (log scale)")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_conditional(tasks: dict, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), layout="constrained")
    fig.suptitle("Conditional coverage in equal-count bins of predicted uncertainty", fontsize=15)
    for ax, task in zip(axes, tasks.values()):
        error = task["error"]
        radius68 = task["sigma"] * task["factors"]["68"]
        radius95 = task["sigma"] * task["factors"]["95"]
        bins = quantile_bins(radius68, 20)
        centers = np.array([np.median(radius68[b]) for b in bins if b.any()])
        cov68 = np.array([np.mean(error[b] <= radius68[b]) for b in bins if b.any()])
        cov95 = np.array([np.mean(error[b] <= radius95[b]) for b in bins if b.any()])
        ax.axhline(0.68, color="#1d5aa6", lw=1.4, ls="--")
        ax.axhline(0.95, color="#cf343d", lw=1.4, ls="--")
        ax.plot(
            centers,
            cov68,
            color="#1d5aa6",
            marker="o",
            ms=3.7,
            lw=2.0,
            label="Observed 68% interval",
        )
        ax.plot(
            centers,
            cov95,
            color="#cf343d",
            marker="s",
            ms=3.5,
            lw=2.0,
            label="Observed 95% interval",
        )
        ax.set(
            xscale="log",
            ylim=(0.4, 1.015),
            title=task["title"],
            xlabel=f"Median predicted 68% radius [{task['unit']}]",
            ylabel="Empirical coverage",
        )
        ax.grid(alpha=0.22, which="both")
    axes[0].legend(loc="lower left", frameon=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def student_radial_quantile(p: np.ndarray, df: float = 3.0):
    return np.sqrt(df * (np.power(1.0 - p, -2.0 / df) - 1.0))


def plot_reliability(tasks: dict, out: Path):
    fig, ax = plt.subplots(figsize=(7.5, 6.8), layout="constrained")
    fig.suptitle("Raw Student-t predictive coverage before validation calibration", fontsize=14)
    probability = np.linspace(0.05, 0.99, 55)
    colors = {"angle": "#205ca8", "line": "#cc3a42", "line200": "#28957a"}
    for key, task in tasks.items():
        ratios = task["error"] / task["sigma"]
        observed = [np.mean(ratios <= q) for q in student_radial_quantile(probability)]
        ax.plot(probability, observed, lw=2.2, color=colors[key], label=task["title"])
    ax.plot([0, 1], [0, 1], color="black", ls="--", lw=1.5, label="Ideal")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Nominal probability", ylabel="Empirical test coverage")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper left", frameon=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--angle-checkpoint", required=True)
    parser.add_argument("--track-checkpoint", required=True)
    parser.add_argument("--uncertainty-checkpoint", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, angle_args = load_base(args.angle_checkpoint, args.track_checkpoint, device)
    model = TrackUncertainty(base).to(device)
    model.head.load_state_dict(
        torch.load(args.uncertainty_checkpoint, map_location="cpu", weights_only=False)["head"]
    )
    test = make_dataset(args, "test", angle_args, None)
    pred = predict(model, make_loader(test, args.workers, False, 42), device)
    calibration = json.loads(Path(args.metrics).read_text())["calibration_from_val"]
    tasks = get_tasks(pred, calibration)
    plot_density(tasks, output / "uncertainty_error_density.png")
    plot_conditional(tasks, output / "uncertainty_conditional_coverage.png")
    plot_reliability(tasks, output / "uncertainty_raw_reliability.png")
    np.savez_compressed(output / "test_predictions.npz", **pred)
    print(
        "Saved plots, predictions, and counts:",
        {k: len(v["error"]) for k, v in tasks.items()},
        flush=True,
    )


if __name__ == "__main__":
    main()
