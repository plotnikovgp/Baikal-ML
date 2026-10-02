#!/usr/bin/env python3
"""Calibrated angle and transverse-track uncertainty for a frozen Baikal model.

The point prediction is exactly the existing DirectionTrackAnchor checkpoint.
Only the small uncertainty head is optimized.  The likelihood is an isotropic
2-D Gaussian in the angular tangent plane and in the transverse track plane.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from train_angle_signal import (
    SetTransformerDirection,
    infinite_batches,
    make_loader,
    seed_everything,
)
from train_track_anchor_v3 import BatchedTrackDataset, DirectionTrackAnchor

RADIAL_68 = math.sqrt(-2 * math.log(1 - 0.68))
RADIAL_95 = math.sqrt(-2 * math.log(1 - 0.95))
SPECIES = {0: "muatm", 1: "nuatm", 2: "nue2"}


def load_base(angle_path: str, track_path: str, device: torch.device):
    angle_checkpoint = torch.load(angle_path, map_location="cpu", weights_only=False)
    angle_args = argparse.Namespace(**angle_checkpoint["args"])
    backbone = SetTransformerDirection(
        angle_args.hidden,
        angle_args.layers,
        angle_args.heads,
        angle_args.ff_mult,
        angle_args.dropout,
        angle_args.feature_mode,
    )
    track_checkpoint = torch.load(track_path, map_location="cpu", weights_only=False)
    scale = float(track_checkpoint["args"].get("anchor_scale_m", 50.0))
    base = DirectionTrackAnchor(backbone, angle_args.hidden, angle_args.dropout, scale)
    base.load_state_dict(track_checkpoint["model"])
    base.requires_grad_(False)
    base.eval()
    return base.to(device), angle_args


class TrackUncertainty(nn.Module):
    def __init__(self, base: DirectionTrackAnchor, hidden: int = 256, dropout: float = 0.05):
        super().__init__()
        self.base = base
        # pooled encoder (2*256), direction, offset, log hits, time spread,
        # and three spatial spreads. No truth or particle label enters the head.
        width = 2 * hidden + 3 + 3 + 1 + 1 + 3
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 2),
        )
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            self.head[-1].bias[:] = torch.tensor([math.log(3.0), math.log(8.0)])

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, x: Tensor, mask: Tensor):
        with torch.no_grad():
            direction, pooled = self.base.backbone(x, mask, return_features=True)
            raw_offset = self.base.point_head(pooled) * self.base.scale_m
            offset = raw_offset - direction * (raw_offset * direction).sum(1, keepdim=True)
            direction = direction.float()
            offset = offset.float()
            pooled = pooled.float()
            count = mask.sum(1).float().clamp_min(1)
            weights = mask.unsqueeze(-1).float()
            mean = (x.float() * weights).sum(1) / count[:, None]
            spread = torch.sqrt(
                (((x.float() - mean[:, None]) ** 2) * weights).sum(1) / count[:, None] + 1e-6
            )
            features = torch.cat(
                [
                    pooled,
                    direction,
                    offset / self.base.scale_m,
                    torch.log(count[:, None]),
                    spread[:, 1:2],
                    spread[:, 2:5],
                ],
                dim=1,
            )
        raw = self.head(features)
        log_angle = raw[:, 0].clamp(math.log(0.1), math.log(90.0))
        log_line = raw[:, 1].clamp(math.log(0.1), math.log(200.0))
        return direction, offset, log_angle, log_line


def residuals(direction: Tensor, offset: Tensor, target_direction: Tensor, target_offset: Tensor):
    # atan2 avoids the precision loss of acos at small angles.
    cross = torch.linalg.cross(direction, target_direction).norm(dim=1)
    dot = (direction * target_direction).sum(1)
    angle = torch.rad2deg(torch.atan2(cross, dot))
    displacement = target_offset - offset
    transverse = displacement - target_direction * (displacement * target_direction).sum(
        1, keepdim=True
    )
    line = transverse.norm(dim=1)
    # At a true-track point 200 m from its centroid-nearest anchor, compute
    # its distance from the reconstructed infinite line.
    point_200 = target_offset + 200.0 * target_direction
    delta_200 = point_200 - offset
    cross_200 = delta_200 - direction * (delta_200 * direction).sum(1, keepdim=True)
    line_200 = cross_200.norm(dim=1)
    return angle, line, line_200


def gaussian_2d_nll(radius: Tensor, log_sigma: Tensor):
    return 2.0 * log_sigma + 0.5 * radius.square() * torch.exp(-2.0 * log_sigma)


def student_2d_nll(radius: Tensor, log_sigma: Tensor, df: float):
    return 2.0 * log_sigma + 0.5 * (df + 2.0) * torch.log1p(
        radius.square() * torch.exp(-2.0 * log_sigma) / df
    )


def numpy_nll(radius, sigma, loss, df):
    ratio2 = (radius / sigma) ** 2
    if loss == "student":
        return 2 * np.log(sigma) + 0.5 * (df + 2.0) * np.log1p(ratio2 / df)
    return 2 * np.log(sigma) + 0.5 * ratio2


def make_dataset(args, split, angle_args, max_events):
    return BatchedTrackDataset(
        args.data,
        split,
        args.batch_size,
        args.max_seq_len,
        bool(getattr(angle_args, "use_signal_norm", False)),
        max_events,
        bool(getattr(angle_args, "center_time", False)),
        anchor_path=args.anchors,
    )


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    chunks = {
        name: []
        for name in ("angle", "line", "line200", "sigma_angle", "sigma_line", "species", "valid")
    }
    for x, target_direction, mask, particle, target_offset, valid in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        target_direction = target_direction.to(device, non_blocking=True)
        target_offset = target_offset.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            direction, offset, log_angle, log_line = model(x, mask)
        angle, line, line200 = residuals(direction, offset, target_direction, target_offset)
        values = dict(
            angle=angle,
            line=line,
            line200=line200,
            sigma_angle=log_angle.exp(),
            sigma_line=log_line.exp(),
            species=particle,
            valid=valid,
        )
        for key, value in values.items():
            chunks[key].append(value.cpu().numpy())
    return {key: np.concatenate(value) for key, value in chunks.items()}


def quantiles(values):
    return {
        "q50": float(np.quantile(values, 0.5)),
        "q68": float(np.quantile(values, 0.68)),
        "q95": float(np.quantile(values, 0.95)),
    }


def calibrate(pred):
    valid = pred["valid"]
    sigma_200 = np.sqrt(
        pred["sigma_line"][valid] ** 2 + (200.0 * np.deg2rad(pred["sigma_angle"][valid])) ** 2
    )

    def radii(error, sigma):
        ratio = error / sigma
        return {"68": float(np.quantile(ratio, 0.68)), "95": float(np.quantile(ratio, 0.95))}

    return {
        "angle": radii(pred["angle"], pred["sigma_angle"]),
        "line": radii(pred["line"][valid], pred["sigma_line"][valid]),
        "line200": radii(pred["line200"][valid], sigma_200),
    }


def summarize_subset(pred, selection, factors):
    selected = np.flatnonzero(selection)
    valid_selection = selected[pred["valid"][selected]]
    if len(selected) == 0 or len(valid_selection) == 0:
        return {"count": int(len(selected)), "valid_track_count": int(len(valid_selection))}
    angle = pred["angle"][selected]
    line = pred["line"][valid_selection]
    line200 = pred["line200"][valid_selection]
    angle_sigma = pred["sigma_angle"][selected]
    line_sigma = pred["sigma_line"][valid_selection]
    angle200_rad = np.deg2rad(pred["sigma_angle"][valid_selection])
    sigma200 = np.sqrt(line_sigma**2 + (200 * angle200_rad) ** 2)

    def coverage(error, sigma, factor):
        return {
            "68": float(np.mean(error <= factor["68"] * sigma)),
            "95": float(np.mean(error <= factor["95"] * sigma)),
        }

    def difficulty_groups(error, sigma):
        edges = np.quantile(sigma, [0, 0.25, 0.5, 0.75, 1])
        groups = []
        for index in range(4):
            group = (sigma >= edges[index]) & (
                sigma <= edges[index + 1] if index == 3 else sigma < edges[index + 1]
            )
            groups.append(
                {
                    "count": int(group.sum()),
                    "error_q68": float(np.quantile(error[group], 0.68)) if group.any() else None,
                }
            )
        return groups

    return {
        "count": int(len(selected)),
        "valid_track_count": int(len(valid_selection)),
        "angle_error_deg": quantiles(angle),
        "angle_r68_deg": quantiles(factors["angle"]["68"] * angle_sigma),
        "angle_r95_deg": quantiles(factors["angle"]["95"] * angle_sigma),
        "angle_coverage": coverage(angle, angle_sigma, factors["angle"]),
        "angle_difficulty_quartiles": difficulty_groups(angle, angle_sigma),
        "line_error_m": quantiles(line),
        "line_r68_m": quantiles(factors["line"]["68"] * line_sigma),
        "line_r95_m": quantiles(factors["line"]["95"] * line_sigma),
        "line_coverage": coverage(line, line_sigma, factors["line"]),
        "line_difficulty_quartiles": difficulty_groups(line, line_sigma),
        "line200_error_m": quantiles(line200),
        "line200_r68_m": quantiles(factors["line200"]["68"] * sigma200),
        "line200_r95_m": quantiles(factors["line200"]["95"] * sigma200),
        "line200_coverage": coverage(line200, sigma200, factors["line200"]),
    }


def report(pred, factors):
    result = {"all": summarize_subset(pred, np.ones(len(pred["angle"]), dtype=bool), factors)}
    for code, name in SPECIES.items():
        result[name] = summarize_subset(pred, pred["species"] == code, factors)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--angle-checkpoint", required=True)
    parser.add_argument("--track-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=("gaussian", "student"), default="gaussian")
    parser.add_argument("--student-df", type=float, default=3.0)
    parser.add_argument("--val-events", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, angle_args = load_base(args.angle_checkpoint, args.track_checkpoint, device)
    model = TrackUncertainty(base).to(device)
    print(
        f"device={device}; trainable={sum(p.numel() for p in model.head.parameters()):,}",
        flush=True,
    )

    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--eval-only requires --checkpoint")
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.head.load_state_dict(saved["head"])
    else:
        train = make_dataset(args, "train", angle_args, None)
        train_batches = infinite_batches(make_loader(train, args.workers, True, args.seed))
        small_val = make_dataset(args, "val", angle_args, args.val_events)
        val_loader = make_loader(small_val, args.workers, False, args.seed)
        optimizer = torch.optim.AdamW(
            model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        best = math.inf
        for step in range(1, args.steps + 1):
            model.train()
            x, truth_dir, mask, _particle, truth_offset, valid = next(train_batches)
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            truth_dir = truth_dir.to(device, non_blocking=True)
            truth_offset = truth_offset.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                direction, offset, log_angle, log_line = model(x, mask)
            angle, line, _ = residuals(direction, offset, truth_dir, truth_offset)
            objective = (
                gaussian_2d_nll
                if args.loss == "gaussian"
                else lambda radius, log_sigma: student_2d_nll(radius, log_sigma, args.student_df)
            )
            loss_angle = objective(angle, log_angle).mean()
            loss_line = objective(line[valid], log_line[valid]).mean()
            loss = loss_angle + loss_line
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at step {step}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.head.parameters(), 2.0)
            optimizer.step()
            if step % 100 == 0:
                print(
                    f"step={step} loss={loss.item():.4f} angle_nll={loss_angle.item():.4f} line_nll={loss_line.item():.4f}",
                    flush=True,
                )
            if step % 300 == 0 or step == args.steps:
                val_pred = predict(model, val_loader, device)
                a = val_pred["angle"]
                line_error = val_pred["line"][val_pred["valid"]]
                sa = val_pred["sigma_angle"]
                sl = val_pred["sigma_line"][val_pred["valid"]]
                val_nll = float(
                    np.mean(numpy_nll(a, sa, args.loss, args.student_df))
                    + np.mean(numpy_nll(line_error, sl, args.loss, args.student_df))
                )
                print(
                    f"validation step={step} nll={val_nll:.5f} calibration={calibrate(val_pred)}",
                    flush=True,
                )
                if val_nll < best:
                    best = val_nll
                    torch.save(
                        {
                            "head": model.head.state_dict(),
                            "args": vars(args),
                            "step": step,
                            "val_nll": val_nll,
                            "base_checkpoint": args.track_checkpoint,
                        },
                        output / "best.pt",
                    )

        saved = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
        model.head.load_state_dict(saved["head"])

    val_full = make_dataset(args, "val", angle_args, None)
    val_pred = predict(model, make_loader(val_full, args.workers, False, args.seed), device)
    # The first val-events were used for selecting the head checkpoint. Keep
    # calibration disjoint from that selection subset and from test.
    calibration_pred = {key: value[args.val_events :] for key, value in val_pred.items()}
    factors = calibrate(calibration_pred)
    print("full validation factors=" + json.dumps(factors), flush=True)
    test = make_dataset(args, "test", angle_args, None)
    test_pred = predict(model, make_loader(test, args.workers, False, args.seed), device)
    result = {
        "calibration_from_val": factors,
        "validation": report(val_pred, factors),
        "test": report(test_pred, factors),
        "checkpoint": str(output / "best.pt" if not args.eval_only else args.checkpoint),
        "note": "Head trained with a radial 2-D Gaussian likelihood. r68/r95 are split-validation empirical conformal radii; line200 uses angle/anchor quadrature before its own calibration. Coverage is marginal on this MC distribution, not guaranteed on experimental data.",
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["test"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
