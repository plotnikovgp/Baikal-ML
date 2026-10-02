#!/usr/bin/env python3
"""Test analytic hit-time direction estimators and neural/analytic blends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from train_angle_signal import BatchedAngleDataset, build_model, make_loader, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=81)
    return parser.parse_args()


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    dots = np.clip(np.sum(prediction * target, axis=1), -1.0, 1.0)
    errors = np.degrees(np.arccos(dots))
    return {
        "count": int(len(errors)),
        "mean": float(errors.mean()),
        "q50": float(np.quantile(errors, 0.50)),
        "q68": float(np.quantile(errors, 0.68)),
        "q90": float(np.quantile(errors, 0.90)),
    }


def weighted_center(
    values: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    denominator = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    mean = (values * weights.unsqueeze(-1)).sum(dim=1) / denominator
    return values - mean.unsqueeze(1), mean


def plane_fit(
    position: torch.Tensor,
    time: torch.Tensor,
    mask: torch.Tensor,
    charge: torch.Tensor,
    charge_weighted: bool,
    robust_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_weights = mask.float()
    if charge_weighted:
        positive_charge = charge.clamp_min(0.0)
        scale = (
            (positive_charge * base_weights).sum(1, keepdim=True)
            / base_weights.sum(1, keepdim=True).clamp_min(1.0)
        ).clamp_min(1e-3)
        base_weights = base_weights * torch.sqrt(positive_charge / scale + 0.05).clamp_max(5.0)
    weights = base_weights
    direction = torch.zeros((len(position), 3), device=position.device)
    residual_rms = torch.zeros(len(position), device=position.device)
    identity = torch.eye(3, device=position.device).unsqueeze(0)
    iterations = max(1, robust_steps + 1)
    for iteration in range(iterations):
        centered_position, position_mean = weighted_center(position, weights)
        centered_time, time_mean = weighted_center(time.unsqueeze(-1), weights)
        centered_time = centered_time.squeeze(-1)
        weighted_position = centered_position * weights.unsqueeze(-1)
        matrix = centered_position.transpose(1, 2) @ weighted_position
        rhs = (weighted_position * centered_time.unsqueeze(-1)).sum(dim=1)
        ridge = 1e-3 * matrix.diagonal(dim1=1, dim2=2).mean(1).clamp_min(1.0)
        gradient = torch.linalg.solve(
            matrix + ridge[:, None, None] * identity, rhs.unsqueeze(-1)
        ).squeeze(-1)
        direction = F.normalize(gradient, dim=1)
        intercept = time_mean.squeeze(-1) - (position_mean * gradient).sum(1)
        residual = time - (position * gradient.unsqueeze(1)).sum(2) - intercept.unsqueeze(1)
        residual_rms = torch.sqrt(
            (residual.square() * base_weights).sum(1) / base_weights.sum(1).clamp_min(1e-6)
        )
        if iteration < robust_steps:
            absolute = residual.abs().masked_fill(~mask, float("nan"))
            scale = torch.nanmedian(absolute, dim=1).values.clamp_min(1.0)
            huber = torch.minimum(
                torch.ones_like(residual), 2.5 * scale.unsqueeze(1) / residual.abs().clamp_min(1e-3)
            )
            weights = base_weights * huber
    return direction, residual_rms


def pca_fit(
    position: torch.Tensor,
    time: torch.Tensor,
    mask: torch.Tensor,
    charge: torch.Tensor,
) -> torch.Tensor:
    weights = mask.float()
    positive_charge = charge.clamp_min(0.0)
    scale = (
        (positive_charge * weights).sum(1, keepdim=True)
        / weights.sum(1, keepdim=True).clamp_min(1.0)
    ).clamp_min(1e-3)
    weights = weights * torch.sqrt(positive_charge / scale + 0.05).clamp_max(5.0)
    centered_position, _ = weighted_center(position, weights)
    centered_time, _ = weighted_center(time.unsqueeze(-1), weights)
    covariance = centered_position.transpose(1, 2) @ (centered_position * weights.unsqueeze(-1))
    eigenvectors = torch.linalg.eigh(covariance).eigenvectors
    axis = eigenvectors[:, :, -1]
    projection = (centered_position * axis.unsqueeze(1)).sum(2)
    time_covariance = (weights * projection * centered_time.squeeze(-1)).sum(1)
    return axis * torch.sign(time_covariance).clamp(min=-1.0, max=1.0).unsqueeze(1)


def best_orientation(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
    positive = metrics(prediction, target)
    negative = metrics(-prediction, target)
    if negative["q68"] < positive["q68"]:
        return -prediction, negative
    return prediction, positive


def blend_search(network: np.ndarray, analytic: np.ndarray, target: np.ndarray) -> dict:
    records = []
    for alpha in np.linspace(-0.25, 1.0, 101):
        prediction = network + float(alpha) * analytic
        prediction /= np.linalg.norm(prediction, axis=1, keepdims=True).clip(min=1e-8)
        value = metrics(prediction, target)
        records.append({"alpha": float(alpha), **value})
    best_q68 = min(records, key=lambda item: (item["q68"], item["q50"]))
    best_q50 = min(records, key=lambda item: (item["q50"], item["q68"]))
    return {"best_q68": best_q68, "best_q50": best_q50}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = SimpleNamespace(**checkpoint["args"])
    model = build_model(model_args).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    center_time = bool(getattr(model_args, "center_time", False))
    if center_time:
        raise ValueError("physical timing fit requires a checkpoint without center_time")
    dataset = BatchedAngleDataset(
        args.data,
        args.split,
        args.batch_size,
        args.max_seq_len,
        bool(getattr(model_args, "use_signal_norm", False)),
        args.max_events,
        center_time,
    )
    loader = make_loader(dataset, args.workers, False, args.seed)
    feature_mean = torch.as_tensor(dataset.target_mean, device=device).view(1, 1, -1)
    feature_std = torch.as_tensor(dataset.target_std, device=device).view(1, 1, -1)
    predictions: dict[str, list[np.ndarray]] = {
        "network": [],
        "plane": [],
        "plane_charge": [],
        "plane_robust": [],
        "pca_charge": [],
    }
    targets = []
    residuals = []
    hit_counts = []
    with torch.inference_mode():
        for x, target, mask, _ in loader:
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            physical = x * feature_std + feature_mean
            charge, time = physical[:, :, 0], physical[:, :, 1]
            position = physical[:, :, 2:5]
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                network = model(x, mask).float()
            plane, _ = plane_fit(position, time, mask, charge, False, 0)
            plane_charge, _ = plane_fit(position, time, mask, charge, True, 0)
            plane_robust, robust_rms = plane_fit(position, time, mask, charge, True, 2)
            pca_charge = pca_fit(position, time, mask, charge)
            for name, value in (
                ("network", network),
                ("plane", plane),
                ("plane_charge", plane_charge),
                ("plane_robust", plane_robust),
                ("pca_charge", pca_charge),
            ):
                predictions[name].append(value.cpu().numpy())
            targets.append(target.numpy())
            residuals.append(robust_rms.cpu().numpy())
            hit_counts.append(mask.sum(1).cpu().numpy())

    prediction_arrays = {name: np.concatenate(parts) for name, parts in predictions.items()}
    target_array = np.concatenate(targets)
    residual_array = np.concatenate(residuals)
    hit_count_array = np.concatenate(hit_counts)
    results: dict[str, object] = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "network": metrics(prediction_arrays["network"], target_array),
        "analytic": {},
        "blends": {},
    }
    oriented: dict[str, np.ndarray] = {}
    for name in ("plane", "plane_charge", "plane_robust", "pca_charge"):
        oriented[name], analytic_metrics = best_orientation(prediction_arrays[name], target_array)
        results["analytic"][name] = analytic_metrics
        results["blends"][name] = blend_search(
            prediction_arrays["network"], oriented[name], target_array
        )
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)
    (output_dir / f"{args.split}_metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(
        output_dir / f"{args.split}_predictions.npz",
        target=target_array,
        residual_rms=residual_array,
        hit_count=hit_count_array,
        **prediction_arrays,
        **{f"oriented_{name}": value for name, value in oriented.items()},
    )


if __name__ == "__main__":
    main()
