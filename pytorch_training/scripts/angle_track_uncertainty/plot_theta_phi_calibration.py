#!/usr/bin/env python3
"""Signed theta/phi uncertainty diagnostics for the frozen direction model.

The uncertainty head predicts one isotropic angular scale, not independent
theta/phi sigmas. We use local spherical geometry (sigma_phi = sigma/sin(theta))
and calibrate each component on held-out validation events before test plots.
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
from train_track_uncertainty_v2 import TrackUncertainty, load_base, make_dataset


@torch.inference_mode()
def component_predictions(model, loader, device, min_sin_theta):
    chunks = {
        name: [] for name in ("theta_error", "phi_error", "sigma_theta", "sigma_phi", "phi_valid")
    }
    model.eval()
    for x, truth_direction, mask, _particle, _anchor, _valid in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        truth_direction = truth_direction.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            predicted_direction, _offset, log_angle, _log_line = model(x, mask)
        prediction = predicted_direction.float()
        truth = truth_direction.float()
        pred_sin = torch.linalg.vector_norm(prediction[:, :2], dim=1)
        truth_sin = torch.linalg.vector_norm(truth[:, :2], dim=1)
        pred_theta = torch.rad2deg(torch.atan2(pred_sin, prediction[:, 2]))
        true_theta = torch.rad2deg(torch.atan2(truth_sin, truth[:, 2]))
        pred_phi = torch.rad2deg(torch.atan2(prediction[:, 1], prediction[:, 0]))
        true_phi = torch.rad2deg(torch.atan2(truth[:, 1], truth[:, 0]))
        theta_error = pred_theta - true_theta
        phi_error = torch.remainder(pred_phi - true_phi + 180.0, 360.0) - 180.0
        sigma_theta = log_angle.float().exp()
        sigma_phi = sigma_theta / pred_sin.clamp_min(min_sin_theta)
        values = dict(
            theta_error=theta_error,
            phi_error=phi_error,
            sigma_theta=sigma_theta,
            sigma_phi=sigma_phi,
            phi_valid=pred_sin >= min_sin_theta,
        )
        for key, value in values.items():
            chunks[key].append(value.cpu().numpy())
    return {key: np.concatenate(value) for key, value in chunks.items()}


def calibration_factors(pred, start):
    result = {}
    for component in ("theta", "phi"):
        valid = np.arange(len(pred["theta_error"])) >= start
        if component == "phi":
            valid &= pred["phi_valid"]
        ratio = np.abs(pred[f"{component}_error"][valid]) / pred[f"sigma_{component}"][valid]
        result[component] = {
            "68": float(np.quantile(ratio, 0.68)),
            "95": float(np.quantile(ratio, 0.95)),
            "calibration_count": int(valid.sum()),
        }
    return result


def equal_count_bins(x, n_bins=24):
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    index = np.searchsorted(edges[1:-1], x, side="right")
    return [index == i for i in range(len(edges) - 1)]


def test_metrics(pred, factors):
    result = {}
    for component in ("theta", "phi"):
        valid = (
            pred["phi_valid"]
            if component == "phi"
            else np.ones(len(pred["theta_error"]), dtype=bool)
        )
        error = pred[f"{component}_error"][valid]
        sigma = pred[f"sigma_{component}"][valid]
        result[component] = {
            "count": int(len(error)),
            "excluded_near_pole": int((~valid).sum()),
            "absolute_error_q50_deg": float(np.quantile(np.abs(error), 0.50)),
            "absolute_error_q68_deg": float(np.quantile(np.abs(error), 0.68)),
            "median_r68_deg": float(np.median(sigma * factors[component]["68"])),
            "median_r95_deg": float(np.median(sigma * factors[component]["95"])),
            "coverage68": float(np.mean(np.abs(error) <= sigma * factors[component]["68"])),
            "coverage95": float(np.mean(np.abs(error) <= sigma * factors[component]["95"])),
        }
    return result


def plot_signed(pred, factors, output):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0), layout="constrained")
    fig.suptitle(
        "Signed angular-component errors versus predicted 68% half-width (nue2 test)", fontsize=15
    )
    for ax, component in zip(axes, ("theta", "phi")):
        valid = (
            pred["phi_valid"]
            if component == "phi"
            else np.ones(len(pred["theta_error"]), dtype=bool)
        )
        error = pred[f"{component}_error"][valid]
        radius = pred[f"sigma_{component}"][valid] * factors[component]["68"]
        limit = max(np.quantile(radius, 0.98), np.quantile(np.abs(error), 0.98))
        shown = (radius <= limit) & (np.abs(error) <= limit)
        hb = ax.hexbin(
            radius[shown],
            error[shown],
            gridsize=65,
            extent=(0, limit, -limit, limit),
            mincnt=1,
            cmap="Blues",
            norm=LogNorm(vmin=1, vmax=3000),
            linewidths=0,
        )
        bins = equal_count_bins(radius)
        centers = np.array([np.median(radius[b]) for b in bins if b.any()])
        lower = np.array([np.quantile(error[b], 0.16) for b in bins if b.any()])
        upper = np.array([np.quantile(error[b], 0.84) for b in bins if b.any()])
        keep = (centers <= limit) & (lower >= -limit) & (upper <= limit)
        ax.plot([0, limit], [0, limit], "k--", lw=1.5, label="Ideal ±r68")
        ax.plot([0, limit], [0, -limit], "k--", lw=1.5)
        ax.plot(
            centers[keep],
            lower[keep],
            color="#d82e35",
            lw=2.3,
            marker="o",
            ms=3,
            label="Observed 16th/84th",
        )
        ax.plot(centers[keep], upper[keep], color="#d82e35", lw=2.3, marker="o", ms=3)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set(
            title=f"{component} (N={len(error):,}; shown={shown.mean():.1%})",
            xlabel="Predicted 68% half-width [deg]",
            ylabel="Predicted - true [deg]",
            xlim=(0, limit),
            ylim=(-limit, limit),
        )
        ax.grid(alpha=0.2)
    axes[0].legend(loc="lower left")
    colorbar = fig.colorbar(hb, ax=axes, shrink=0.84, pad=0.01)
    colorbar.set_label("Events per hexbin (log scale)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_conditional(pred, factors, output):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3), layout="constrained")
    fig.suptitle("Conditional theta/phi interval coverage by predicted uncertainty", fontsize=15)
    for ax, component in zip(axes, ("theta", "phi")):
        valid = (
            pred["phi_valid"]
            if component == "phi"
            else np.ones(len(pred["theta_error"]), dtype=bool)
        )
        error = np.abs(pred[f"{component}_error"][valid])
        sigma = pred[f"sigma_{component}"][valid]
        r68 = sigma * factors[component]["68"]
        r95 = sigma * factors[component]["95"]
        bins = equal_count_bins(r68, 20)
        x = np.array([np.median(r68[b]) for b in bins if b.any()])
        c68 = np.array([np.mean(error[b] <= r68[b]) for b in bins if b.any()])
        c95 = np.array([np.mean(error[b] <= r95[b]) for b in bins if b.any()])
        ax.axhline(0.68, color="#1d5aa6", ls="--", lw=1.4)
        ax.axhline(0.95, color="#cf343d", ls="--", lw=1.4)
        ax.plot(x, c68, color="#1d5aa6", marker="o", ms=3.5, lw=2, label="Observed 68%")
        ax.plot(x, c95, color="#cf343d", marker="s", ms=3.5, lw=2, label="Observed 95%")
        ax.set(
            title=component,
            xlabel="Median predicted 68% half-width [deg]",
            ylabel="Empirical coverage",
            xscale="log",
            ylim=(0.35, 1.015),
        )
        ax.grid(alpha=0.2, which="both")
    axes[0].legend(loc="lower left")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--angle-checkpoint", required=True)
    parser.add_argument("--track-checkpoint", required=True)
    parser.add_argument("--uncertainty-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--selection-events", type=int, default=50_000)
    parser.add_argument("--min-sin-theta", type=float, default=0.15)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, angle_args = load_base(args.angle_checkpoint, args.track_checkpoint, device)
    model = TrackUncertainty(base).to(device)
    model.head.load_state_dict(
        torch.load(args.uncertainty_checkpoint, map_location="cpu", weights_only=False)["head"]
    )
    val = make_dataset(args, "val", angle_args, None)
    val_predictions = component_predictions(
        model, make_loader(val, args.workers, False, 42), device, args.min_sin_theta
    )
    factors = calibration_factors(val_predictions, args.selection_events)
    print("validation calibration=" + json.dumps(factors), flush=True)
    test = make_dataset(args, "test", angle_args, None)
    test_predictions = component_predictions(
        model, make_loader(test, args.workers, False, 42), device, args.min_sin_theta
    )
    metrics = {
        "min_sin_theta": args.min_sin_theta,
        "calibration_from_val": factors,
        "test": test_metrics(test_predictions, factors),
    }
    (output / "theta_phi_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(output / "theta_phi_test_predictions.npz", **test_predictions)
    plot_signed(test_predictions, factors, output / "theta_phi_signed_calibration.png")
    plot_conditional(test_predictions, factors, output / "theta_phi_conditional_coverage.png")
    print(json.dumps(metrics["test"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
