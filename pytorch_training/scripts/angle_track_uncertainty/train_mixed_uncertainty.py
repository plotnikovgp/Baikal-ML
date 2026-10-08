#!/usr/bin/env python3
"""Fit one uncertainty head for the single-muatm universal direction model.

The base direction/track model is frozen. Predicted-hit and GT-hit batches use
the same input normalization, while their disjoint validation halves provide
separate post-training coverage factors. Production uses predicted-hit factors.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from train_angle_signal import infinite_batches, make_loader, seed_everything
from train_mixed_universal import AdaptedTrackDataset, next_nonempty
from train_track_uncertainty_v2 import (
    TrackUncertainty,
    calibrate,
    gaussian_2d_nll,
    load_base,
    numpy_nll,
    report,
    residuals,
    student_2d_nll,
)


@torch.inference_mode()
def predict_nonempty(model, loader, device):
    model.eval()
    chunks = {
        name: []
        for name in (
            "angle",
            "line",
            "line200",
            "sigma_angle",
            "sigma_line",
            "species",
            "valid",
        )
    }
    for x, target_direction, mask, particle, target_offset, valid in loader:
        if not len(x):
            continue
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        target_direction = target_direction.to(device, non_blocking=True)
        target_offset = target_offset.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            direction, offset, log_angle, log_line = model(x, mask)
        angle, line, line200 = residuals(
            direction.float(), offset.float(), target_direction, target_offset
        )
        values = {
            "angle": angle,
            "line": line,
            "line200": line200,
            "sigma_angle": log_angle.float().exp(),
            "sigma_line": log_line.float().exp(),
            "species": particle,
            "valid": valid,
        }
        for key, value in values.items():
            chunks[key].append(value.cpu().numpy())
    return {key: np.concatenate(value) for key, value in chunks.items()}


def validation_nll(pred, loss: str, student_df: float) -> float:
    valid = pred["valid"]
    return float(
        np.mean(numpy_nll(pred["angle"], pred["sigma_angle"], loss, student_df))
        + np.mean(
            numpy_nll(
                pred["line"][valid],
                pred["sigma_line"][valid],
                loss,
                student_df,
            )
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-data", type=Path, required=True)
    parser.add_argument("--pred-anchors", type=Path, required=True)
    parser.add_argument("--gt-data", type=Path, required=True)
    parser.add_argument("--gt-anchors", type=Path, required=True)
    parser.add_argument("--norm-data", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--angle-checkpoint", type=Path, required=True)
    parser.add_argument("--track-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--val-every", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gt-weight", type=float, default=0.5)
    parser.add_argument("--loss", choices=("gaussian", "student"), default="student")
    parser.add_argument("--student-df", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=110)
    parser.add_argument("--init-head", type=Path, default=None)
    args = parser.parse_args()
    if not 0 <= args.gt_weight <= 1:
        raise ValueError("gt-weight must be between zero and one")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n"
    )
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, _ = load_base(str(args.angle_checkpoint), str(args.track_checkpoint), device)
    model = TrackUncertainty(base).to(device)
    if args.init_head is not None:
        initialized = torch.load(args.init_head, map_location="cpu", weights_only=False)
        if Path(initialized["base_checkpoint"]).resolve() != args.track_checkpoint.resolve():
            raise ValueError("Initial uncertainty head was trained on a different base")
        model.head.load_state_dict(initialized["head"], strict=True)
    with h5py.File(args.norm_data, "r") as h5:
        expected_mean = np.asarray(h5["norm_param/mean"], dtype=np.float32)
        expected_std = np.asarray(h5["norm_param/std"], dtype=np.float32)
    with np.load(args.masks) as masks_file:
        masks = {name: masks_file[name].copy() for name in masks_file.files}

    def loader(path, anchors, split, keep, single_mu, shuffle=False, seed=0):
        dataset = AdaptedTrackDataset(
            str(path),
            split,
            args.batch_size,
            args.max_seq_len,
            False,
            None,
            False,
            anchor_path=str(anchors),
            keep=keep,
            expected_mean=expected_mean,
            expected_std=expected_std,
            single_muatm_only=single_mu,
        )
        return make_loader(dataset, args.workers, shuffle, seed)

    pred_train = infinite_batches(
        loader(args.pred_data, args.pred_anchors, "train", None, True, True, args.seed)
    )
    gt_train = infinite_batches(
        loader(
            args.gt_data,
            args.gt_anchors,
            "train",
            masks["gt_train"],
            False,
            True,
            args.seed + 1,
        )
    )
    pred_val_index = np.arange(len(masks["pred_val"]))
    gt_val_index = np.arange(len(masks["gt_val"]))
    pred_select = masks["pred_val"] & (pred_val_index % 2 == 0)
    gt_select = masks["gt_val"] & (gt_val_index % 6 == 0)
    pred_calibrate = masks["pred_val"] & (pred_val_index % 2 == 1)
    gt_calibrate = masks["gt_val"] & (gt_val_index % 2 == 1)
    pred_val = loader(args.pred_data, args.pred_anchors, "val", pred_select, True)
    gt_val = loader(args.gt_data, args.gt_anchors, "val", gt_select, False)

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best = math.inf
    best_path = args.output_dir / "best.pt"
    if args.init_head is not None:
        pred_initial = predict_nonempty(model, pred_val, device)
        gt_initial = predict_nonempty(model, gt_val, device)
        pred_nll = validation_nll(pred_initial, args.loss, args.student_df)
        gt_nll = validation_nll(gt_initial, args.loss, args.student_df)
        best = (1 - args.gt_weight) * pred_nll + args.gt_weight * gt_nll
        torch.save(
            {
                "head": model.head.state_dict(),
                "args": vars(args),
                "step": 0,
                "val_nll": best,
                "base_checkpoint": str(args.track_checkpoint),
            },
            best_path,
        )
        print(f"initial validation score={best:.5f}", flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = {}
        for name, source, weight in (
            ("pred", pred_train, 1 - args.gt_weight),
            ("gt", gt_train, args.gt_weight),
        ):
            x, truth_dir, mask, _particle, truth_offset, valid = next_nonempty(source)
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            truth_dir = truth_dir.to(device, non_blocking=True)
            truth_offset = truth_offset.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                direction, offset, log_angle, log_line = model(x, mask)
            angle, line, _ = residuals(direction.float(), offset.float(), truth_dir, truth_offset)
            objective = (
                gaussian_2d_nll
                if args.loss == "gaussian"
                else lambda radius, log_sigma: student_2d_nll(radius, log_sigma, args.student_df)
            )
            angle_nll = objective(angle, log_angle.float()).mean()
            line_nll = (
                objective(line[valid], log_line[valid].float()).mean()
                if torch.any(valid)
                else angle_nll * 0.0
            )
            loss = weight * (angle_nll + line_nll)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step} ({name})")
            scaler.scale(loss).backward()
            losses[name] = float(loss.detach())
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.head.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        if step % 100 == 0:
            print(f"step={step} losses={losses}", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            pred_result = predict_nonempty(model, pred_val, device)
            gt_result = predict_nonempty(model, gt_val, device)
            pred_nll = validation_nll(pred_result, args.loss, args.student_df)
            gt_nll = validation_nll(gt_result, args.loss, args.student_df)
            score = (1 - args.gt_weight) * pred_nll + args.gt_weight * gt_nll
            metrics = {
                "step": step,
                "pred_count": int(len(pred_result["angle"])),
                "gt_count": int(len(gt_result["angle"])),
                "pred_nll": pred_nll,
                "gt_nll": gt_nll,
                "score": score,
            }
            print("validation " + json.dumps(metrics), flush=True)
            (args.output_dir / f"val_step_{step}.json").write_text(
                json.dumps(metrics, indent=2) + "\n"
            )
            if score < best:
                best = score
                torch.save(
                    {
                        "head": model.head.state_dict(),
                        "args": vars(args),
                        "step": step,
                        "val_nll": score,
                        "base_checkpoint": str(args.track_checkpoint),
                    },
                    best_path,
                )

    if not best_path.exists():
        raise ValueError("No uncertainty checkpoint was produced")
    saved = torch.load(best_path, map_location="cpu", weights_only=False)
    model.head.load_state_dict(saved["head"])
    pred_cal = predict_nonempty(
        model, loader(args.pred_data, args.pred_anchors, "val", pred_calibrate, True), device
    )
    gt_cal = predict_nonempty(
        model, loader(args.gt_data, args.gt_anchors, "val", gt_calibrate, False), device
    )
    pred_factors = calibrate(pred_cal)
    gt_factors = calibrate(gt_cal)
    pred_test = predict_nonempty(
        model, loader(args.pred_data, args.pred_anchors, "test", masks["pred_test"], True), device
    )
    gt_test = predict_nonempty(
        model, loader(args.gt_data, args.gt_anchors, "test", masks["gt_test"], False), device
    )
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        **{f"pred_{key}": value for key, value in pred_test.items()},
        **{f"gt_{key}": value for key, value in gt_test.items()},
    )
    results = {
        "calibration_from_val": pred_factors,
        "calibration_gt_from_val": gt_factors,
        "calibration_counts": {
            "pred": int(len(pred_cal["angle"])),
            "gt": int(len(gt_cal["angle"])),
        },
        "test": {
            "predicted_hits_single_muatm": report(pred_test, pred_factors),
            "gt_nue2": report(gt_test, gt_factors),
        },
        "checkpoint": str(best_path),
        "note": (
            "Independent event-ID masks; muatm kept only when original MC has "
            "exactly one simulated track and a valid anchor. Calibration halves "
            "are disjoint from head-selection validation. No experimental-data "
            "coverage is claimed."
        ),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(results["test"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
