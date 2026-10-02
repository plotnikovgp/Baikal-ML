#!/usr/bin/env python3
"""Selective quality and conditional calibration on saved independent-test predictions.

The angle error is the geodesic direction error. The track-point error is the
transverse displacement of the predicted anchor from the *true* track line;
it is not a displacement evaluated at an arbitrary lever arm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RETENTION = np.linspace(0.1, 1.0, 19)
DECILES = 10
COLOR_SELECTED = "#2258a5"
COLOR_ORACLE = "#18836b"
COLOR_BASELINE = "#555b64"
COLOR_68 = "#2258a5"
COLOR_95 = "#c4484a"


def quality_curve(error: np.ndarray, score: np.ndarray) -> dict:
    order = np.argsort(score, kind="stable")
    oracle_order = np.argsort(error, kind="stable")
    counts = np.maximum(1, np.ceil(RETENTION * len(error)).astype(int))
    return {
        "retention": RETENTION,
        "selected_q50": np.array([np.quantile(error[order[:n]], 0.5) for n in counts]),
        "selected_q68": np.array([np.quantile(error[order[:n]], 0.68) for n in counts]),
        "selected_q95": np.array([np.quantile(error[order[:n]], 0.95) for n in counts]),
        "oracle_q68": np.array([np.quantile(error[oracle_order[:n]], 0.68) for n in counts]),
    }


def conditional_coverage(error: np.ndarray, sigma: np.ndarray, factors: dict) -> dict:
    order = np.argsort(sigma, kind="stable")
    chunks = np.array_split(order, DECILES)
    r68 = sigma * factors["68"]
    r95 = sigma * factors["95"]
    return {
        "decile": np.arange(1, DECILES + 1),
        "median_r68": np.array([np.median(r68[c]) for c in chunks]),
        "coverage68": np.array([np.mean(error[c] <= r68[c]) for c in chunks]),
        "coverage95": np.array([np.mean(error[c] <= r95[c]) for c in chunks]),
        "count": np.array([len(c) for c in chunks]),
    }


def build_task(error: np.ndarray, sigma: np.ndarray, factors: dict) -> dict:
    keep = np.isfinite(error) & np.isfinite(sigma) & (sigma > 0)
    error = error[keep].astype(np.float64)
    sigma = sigma[keep].astype(np.float64)
    if not len(error):
        raise ValueError("No finite errors and positive uncertainty predictions")
    return {
        "error": error,
        "sigma": sigma,
        "quality": quality_curve(error, sigma),
        "calibration": conditional_coverage(error, sigma, factors),
        "factors": factors,
    }


def draw(tasks: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.2), layout="constrained")
    fig.suptitle("Uncertainty usefulness and calibration · independent nue2 test", fontsize=15)
    for col, (title, task) in enumerate(tasks.items()):
        error = task["error"]
        q = task["quality"]
        c = task["calibration"]
        unit = "°" if col == 0 else "m"
        ax = axes[0, col]
        x = q["retention"] * 100
        ax.plot(
            x,
            q["selected_q68"],
            lw=2.4,
            color=COLOR_SELECTED,
            label="Selected by predicted uncertainty",
        )
        ax.plot(
            x,
            q["oracle_q68"],
            lw=1.8,
            ls=":",
            color=COLOR_ORACLE,
            label="Oracle (true error; unattainable)",
        )
        ax.axhline(
            np.quantile(error, 0.68), lw=1.6, ls="--", color=COLOR_BASELINE, label="No selection"
        )
        ax.set(
            title=title,
            xlabel="Retained events with lowest predicted uncertainty [%]",
            ylabel=f"Direction q68 [{unit}]" if col == 0 else f"Transverse anchor q68 [{unit}]",
            xlim=(10, 100),
            ylim=(0, None),
        )
        ax.grid(alpha=0.24)
        ax.legend(loc="upper left", fontsize=8.5)
        ax.text(
            0.98,
            0.04,
            f"N={len(error):,}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )

        ax = axes[1, col]
        ax.plot(
            c["decile"],
            c["coverage68"] * 100,
            marker="o",
            ms=4.5,
            lw=2,
            color=COLOR_68,
            label="Observed 68% radius",
        )
        ax.plot(
            c["decile"],
            c["coverage95"] * 100,
            marker="s",
            ms=4,
            lw=2,
            color=COLOR_95,
            label="Observed 95% radius",
        )
        ax.axhline(68, lw=1.5, ls="--", color=COLOR_68, alpha=0.65)
        ax.axhline(95, lw=1.5, ls="--", color=COLOR_95, alpha=0.65)
        ax.set(
            xlabel="Predicted-uncertainty decile (easy → hard)",
            ylabel="Actual fraction inside predicted radius [%]",
            xlim=(0.8, 10.2),
            ylim=(35, 102),
            xticks=np.arange(1, 11),
        )
        ax.grid(alpha=0.24)
        ax.legend(loc="lower left", fontsize=8.5)
    fig.savefig(output, dpi=185)
    plt.close(fig)


def summarized(task: dict) -> dict:
    q = task["quality"]
    c = task["calibration"]
    error = task["error"]
    retentions = {}
    for fraction in (0.5, 0.8, 1.0):
        ix = int(np.argmin(abs(q["retention"] - fraction)))
        retentions[str(fraction)] = {
            "q50": float(q["selected_q50"][ix]),
            "q68": float(q["selected_q68"][ix]),
            "q95": float(q["selected_q95"][ix]),
            "oracle_q68": float(q["oracle_q68"][ix]),
        }
    return {
        "count": len(error),
        "retained": retentions,
        "marginal_coverage68": float(np.mean(error <= task["sigma"] * task["factors"]["68"])),
        "marginal_coverage95": float(np.mean(error <= task["sigma"] * task["factors"]["95"])),
        "conditional_coverage68_by_decile": c["coverage68"].tolist(),
        "conditional_coverage95_by_decile": c["coverage95"].tolist(),
        "median_predicted_r68_by_decile": c["median_r68"].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.predictions) as pred:
        calibration = json.loads(args.metrics.read_text())["calibration_from_val"]
        angle = build_task(pred["angle"], pred["sigma_angle"], calibration["angle"])
        valid = pred["valid"].astype(bool)
        track = build_task(pred["line"][valid], pred["sigma_line"][valid], calibration["line"])
        joint = {
            "both_68": float(
                np.mean(
                    (
                        pred["angle"][valid]
                        <= pred["sigma_angle"][valid] * calibration["angle"]["68"]
                    )
                    & (pred["line"][valid] <= pred["sigma_line"][valid] * calibration["line"]["68"])
                )
            ),
            "both_95": float(
                np.mean(
                    (
                        pred["angle"][valid]
                        <= pred["sigma_angle"][valid] * calibration["angle"]["95"]
                    )
                    & (pred["line"][valid] <= pred["sigma_line"][valid] * calibration["line"]["95"])
                )
            ),
        }
    tasks = {"Direction": angle, "Reference-point transverse distance": track}
    draw(tasks, args.output_dir / "quality_calibration.png")
    summary = {
        "definition": "angle=geodesic direction error in deg; line=predicted anchor transverse distance to true line in m",
        "split": "independent nue2 test; calibration factors fit on validation",
        "angle_deg": summarized(angle),
        "anchor_transverse_m": summarized(track),
        "joint_coverage": joint,
    }
    out = args.output_dir / "quality_calibration_metrics.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
