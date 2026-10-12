#!/usr/bin/env python3
"""Validation-only diagnostic for analytic space-time track directions.

This script deliberately does not fit a deployable correction or inspect test.
It measures whether physical hit geometry contains complementary information
to the selected v4 network on the same leakage-free predicted-hit validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from train_mixed_universal import AdaptedTrackDataset, model_from_checkpoints


def direction_error(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    dot = (prediction * truth).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(dot))


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    with h5py.File(args.norm_data, "r") as h5:
        mean = np.asarray(h5["norm_param/mean"], dtype=np.float32)
        std = np.asarray(h5["norm_param/std"], dtype=np.float32)
    with np.load(args.masks) as masks:
        keep = masks["pred_val"].copy()
    data = AdaptedTrackDataset(
        str(args.pred_data),
        "val",
        args.batch_size,
        128,
        False,
        None,
        False,
        anchor_path=str(args.pred_anchors),
        keep=keep,
        expected_mean=mean,
        expected_std=std,
        single_muatm_only=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_from_checkpoints(args.angle_checkpoint, args.track_checkpoint, 50.0, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    feature_mean = torch.from_numpy(mean).to(device)
    feature_std = torch.from_numpy(std).to(device)

    chunks: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "nn",
            "pca_forward",
            "pca_reverse",
            "time_gradient",
            "covariance_direction",
            "axis_ratio",
            "time_correlation",
            "particle",
            "hits",
            "span_m",
        )
    }
    for index in range(len(data)):
        x, truth, mask, particle, _point, _valid = data[index]
        if not len(x):
            continue
        x, truth, mask = x.to(device), truth.to(device), mask.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            nn_dir, _point = model(x, mask)
        nn_dir = nn_dir.float()
        raw = x[..., :5] * feature_std + feature_mean
        weights = mask.float()
        count = weights.sum(dim=1).clamp_min(1.0)
        pos = raw[..., 2:5]
        time = raw[..., 1]
        center_pos = (pos * weights[..., None]).sum(dim=1) / count[:, None]
        center_time = (time * weights).sum(dim=1) / count
        pos = (pos - center_pos[:, None]) * weights[..., None]
        time = (time - center_time[:, None]) * weights
        covariance = torch.einsum("bni,bnj->bij", pos, pos) / count[:, None, None]
        time_cov = torch.einsum("bni,bn->bi", pos, time) / count[:, None]
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        axis = eigenvectors[..., -1]
        axis_sign = torch.sign((axis * time_cov).sum(dim=1)).clamp(min=-1, max=1)
        axis_sign = torch.where(axis_sign == 0, torch.ones_like(axis_sign), axis_sign)
        axis = axis * axis_sign[:, None]
        regularized = covariance + torch.eye(3, device=device)[None] * 625.0
        gradient = torch.linalg.solve(regularized, time_cov[..., None]).squeeze(-1)
        gradient = F.normalize(gradient, dim=1)
        cov_dir = F.normalize(time_cov, dim=1)
        ratio = eigenvalues[:, -1] / eigenvalues[:, -2].clamp_min(1.0)
        axis_proj = (pos * axis[:, None]).sum(dim=-1)
        corr = (axis_proj * time).sum(dim=1) / (
            torch.linalg.vector_norm(axis_proj, dim=1) * torch.linalg.vector_norm(time, dim=1)
        ).clamp_min(1e-6)
        span = (
            torch.cdist(pos, pos)
            .masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0.0)
            .amax(dim=(1, 2))
        )
        batch = {
            "nn": direction_error(nn_dir, truth),
            "pca_forward": direction_error(axis, truth),
            "pca_reverse": direction_error(-axis, truth),
            "time_gradient": direction_error(gradient, truth),
            "covariance_direction": direction_error(cov_dir, truth),
            "axis_ratio": ratio,
            "time_correlation": corr,
            "particle": particle,
            "hits": count,
            "span_m": span,
        }
        for name, value in batch.items():
            chunks[name].append(value.detach().cpu().numpy())

    arrays = {name: np.concatenate(parts) for name, parts in chunks.items()}
    summary = {"events": int(len(arrays["nn"]))}
    metrics = (
        "nn",
        "pca_forward",
        "pca_reverse",
        "time_gradient",
        "covariance_direction",
    )
    for subset, selection in (
        ("all", np.ones(len(arrays["nn"]), dtype=bool)),
        ("nuatm", arrays["particle"] == 1),
        ("single_muatm", arrays["particle"] == 0),
        ("axis_ratio_ge_4", arrays["axis_ratio"] >= 4),
        ("axis_ratio_ge_10", arrays["axis_ratio"] >= 10),
        ("time_corr_ge_0.8", arrays["time_correlation"] >= 0.8),
        ("span_ge_300m", arrays["span_m"] >= 300),
    ):
        if not selection.any():
            continue
        summary[subset] = {"count": int(selection.sum())}
        for name in metrics:
            values = arrays[name][selection]
            summary[subset][name] = {
                "q50": float(np.quantile(values, 0.5)),
                "q68": float(np.quantile(values, 0.68)),
            }
        summary[subset]["pca_oracle_better_fraction"] = float(
            np.mean(arrays["pca_forward"][selection] < arrays["nn"][selection])
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-data", type=Path, required=True)
    parser.add_argument("--pred-anchors", type=Path, required=True)
    parser.add_argument("--norm-data", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--angle-checkpoint", type=Path, required=True)
    parser.add_argument("--track-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
