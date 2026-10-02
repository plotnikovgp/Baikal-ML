#!/usr/bin/env python3
"""Evaluate reconstruction and MC/experiment angular alignment on full splits."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from domain_adapt_angle_signal import (
    UnlabeledAngleDataset,
    alignment_metrics,
    predict_labeled,
    predict_unlabeled,
)
from train_angle_signal import (
    BatchedAngleDataset,
    build_model,
    evaluate,
    make_loader,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--adapted-checkpoint", required=True)
    parser.add_argument("--truth-data", required=True)
    parser.add_argument("--matched-mc-data", required=True)
    parser.add_argument("--experimental-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--truth-batch-size", type=int, default=1024)
    parser.add_argument("--domain-batch-size", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--rotation-tta", action="store_true")
    return parser.parse_args()


def checkpoint_model_args(checkpoint: dict) -> SimpleNamespace:
    values = checkpoint.get("initial_model_args", checkpoint.get("args"))
    if values is None:
        raise KeyError("checkpoint has neither 'args' nor 'initial_model_args'")
    return SimpleNamespace(**values)


def load_model(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu")
    model_args = checkpoint_model_args(checkpoint)
    model = build_model(model_args).to(device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()
    return model, model_args


def direction_angles(prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.degrees(np.arccos(np.clip(prediction[:, 2], -1.0, 1.0)))
    phi = np.mod(np.degrees(np.arctan2(prediction[:, 1], prediction[:, 0])), 360.0)
    return theta, phi


def rotate_xy(values: torch.Tensor, angle: float) -> torch.Tensor:
    cosine, sine = math.cos(angle), math.sin(angle)
    result = values.clone()
    result[..., 0] = cosine * values[..., 0] - sine * values[..., 1]
    result[..., 1] = sine * values[..., 0] + cosine * values[..., 1]
    return result


def rotation_tta_prediction(
    model,
    x: torch.Tensor,
    mask: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> torch.Tensor:
    physical = x * feature_std + feature_mean
    predictions = []
    for angle in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
        rotated_physical = physical.clone()
        rotated_physical[:, :, 2:4] = rotate_xy(physical[:, :, 2:4], angle)
        rotated_x = (rotated_physical - feature_mean) / feature_std
        rotated_x = torch.where(mask.unsqueeze(-1), rotated_x, torch.zeros_like(rotated_x))
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=x.device.type == "cuda"
        ):
            prediction = model(rotated_x, mask).float()
        prediction[:, :2] = rotate_xy(prediction[:, :2], -angle)
        predictions.append(prediction)
    return F.normalize(torch.stack(predictions).mean(0), dim=1)


def metrics_from_arrays(
    prediction: np.ndarray,
    target: np.ndarray,
    particle: np.ndarray,
) -> dict[str, float]:
    errors = np.degrees(np.arccos(np.clip(np.sum(prediction * target, axis=1), -1.0, 1.0)))
    result = {
        "count": int(len(errors)),
        "mean": float(errors.mean()),
        "q50": float(np.quantile(errors, 0.50)),
        "q68": float(np.quantile(errors, 0.68)),
        "q90": float(np.quantile(errors, 0.90)),
    }
    for index, name in enumerate(("muatm", "nuatm", "nue2")):
        selected = errors[particle == index]
        if len(selected):
            result[f"{name}_count"] = int(len(selected))
            result[f"{name}_q50"] = float(np.quantile(selected, 0.50))
            result[f"{name}_q68"] = float(np.quantile(selected, 0.68))
            result[f"{name}_q90"] = float(np.quantile(selected, 0.90))
    return result


@torch.inference_mode()
def predict_labeled_tta(model, loader, device, feature_mean, feature_std):
    model.eval()
    predictions, targets, particles = [], [], []
    mean = torch.as_tensor(feature_mean, device=device).view(1, 1, -1)
    std = torch.as_tensor(feature_std, device=device).view(1, 1, -1)
    for x, target, mask, particle in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        predictions.append(rotation_tta_prediction(model, x, mask, mean, std).cpu().numpy())
        targets.append(target.numpy())
        particles.append(particle.numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    particle = np.concatenate(particles)
    return prediction, target, particle


@torch.inference_mode()
def predict_unlabeled_tta(model, loader, device, feature_mean, feature_std):
    model.eval()
    predictions = []
    mean = torch.as_tensor(feature_mean, device=device).view(1, 1, -1)
    std = torch.as_tensor(feature_std, device=device).view(1, 1, -1)
    for x, mask in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        predictions.append(rotation_tta_prediction(model, x, mask, mean, std).cpu().numpy())
    return np.concatenate(predictions)


def make_plot(predictions: dict[str, np.ndarray], metrics: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    theta_bins = np.linspace(0.0, 180.0, 61)
    phi_bins = np.linspace(0.0, 360.0, 49)
    for column, stage in enumerate(("baseline", "adapted")):
        mc_theta, mc_phi = direction_angles(predictions[f"{stage}_mc"])
        exp_theta, exp_phi = direction_angles(predictions[f"{stage}_exp"])
        axes[0, column].hist(
            mc_theta, bins=theta_bins, density=True, histtype="step", lw=2, label="muatm MC"
        )
        axes[0, column].hist(
            exp_theta, bins=theta_bins, density=True, histtype="step", lw=2, label="experiment"
        )
        axes[0, column].set(
            xlabel=r"predicted $\theta$ [deg]", ylabel="density", title=stage.capitalize()
        )
        axes[0, column].legend(loc="upper right")
        axes[1, column].hist(
            mc_phi, bins=phi_bins, density=True, histtype="step", lw=2, label="muatm MC"
        )
        axes[1, column].hist(
            exp_phi, bins=phi_bins, density=True, histtype="step", lw=2, label="experiment"
        )
        axes[1, column].set(xlabel=r"predicted $\phi$ [deg]", ylabel="density")
        alignment = metrics[stage]["alignment"]
        axes[0, column].text(
            0.03,
            0.96,
            f"SWD={alignment['direction_swd']:.4f}\n"
            f"W(theta)={alignment['theta_wasserstein_deg']:.2f} deg\n"
            f"KS(theta)={alignment['theta_ks']:.3f}",
            transform=axes[0, column].transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
    fig.suptitle("Predicted-angle domain comparison after identical signal selection")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline_model, baseline_args = load_model(args.baseline_checkpoint, device)
    adapted_model, adapted_args = load_model(args.adapted_checkpoint, device)
    baseline_center = bool(getattr(baseline_args, "center_time", False))
    adapted_center = bool(getattr(adapted_args, "center_time", False))
    if baseline_center != adapted_center:
        raise ValueError("baseline and adapted checkpoints use different center_time settings")
    center_time = baseline_center

    truth_dataset = BatchedAngleDataset(
        args.truth_data,
        args.split,
        args.truth_batch_size,
        args.max_seq_len,
        bool(getattr(baseline_args, "use_signal_norm", False)),
        None,
        center_time,
    )
    mc_dataset = BatchedAngleDataset(
        args.matched_mc_data,
        args.split,
        args.domain_batch_size,
        args.max_seq_len,
        False,
        None,
        center_time,
    )
    exp_dataset = UnlabeledAngleDataset(
        args.experimental_data,
        args.split,
        args.domain_batch_size,
        args.max_seq_len,
        center_time,
    )
    truth_loader = make_loader(truth_dataset, args.workers, False, args.seed)
    mc_loader = make_loader(mc_dataset, args.workers, False, args.seed + 1)
    exp_loader = make_loader(exp_dataset, args.workers, False, args.seed + 2)

    predictions: dict[str, np.ndarray] = {}
    results: dict[str, dict] = {"split": args.split, "device": str(device)}
    for stage, model in (("baseline", baseline_model), ("adapted", adapted_model)):
        if args.rotation_tta:
            truth_prediction, truth_target, truth_particle = predict_labeled_tta(
                model, truth_loader, device, truth_dataset.target_mean, truth_dataset.target_std
            )
            mc_prediction, mc_target, mc_particle = predict_labeled_tta(
                model, mc_loader, device, mc_dataset.target_mean, mc_dataset.target_std
            )
            exp_prediction = predict_unlabeled_tta(
                model, exp_loader, device, exp_dataset.feature_mean, exp_dataset.feature_std
            )
            truth_metrics = metrics_from_arrays(truth_prediction, truth_target, truth_particle)
            matched_metrics = metrics_from_arrays(mc_prediction, mc_target, mc_particle)
        else:
            truth_metrics = evaluate(model, truth_loader, device)
            matched_metrics = evaluate(model, mc_loader, device)
            mc_prediction = predict_labeled(model, mc_loader, device)
            exp_prediction = predict_unlabeled(model, exp_loader, device)
        predictions[f"{stage}_mc"] = mc_prediction
        predictions[f"{stage}_exp"] = exp_prediction
        results[stage] = {
            "truth": truth_metrics,
            "matched_mc": matched_metrics,
            "alignment": alignment_metrics(mc_prediction, exp_prediction),
        }
        print(stage + " " + json.dumps(results[stage], sort_keys=True), flush=True)

    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(output_dir / "predictions.npz", **predictions)
    make_plot(predictions, results, output_dir / "angle_distribution_comparison.png")


if __name__ == "__main__":
    main()
