#!/usr/bin/env python3
"""Fine-tune one v4 direction/point model with extra weight on real short-hit events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from bundle_universal_model import build_model
from train_angle_signal import infinite_batches, loss_function, seed_everything
from train_mixed_universal import evaluate, next_nonempty
from train_photon_residual import data_loader
from train_track_anchor_v3 import point_loss


def hit_span_m(x: torch.Tensor, mask: torch.Tensor, mean, std) -> torch.Tensor:
    """Maximum selected-hit separation in physical metres, with padding excluded."""
    xyz = x[..., 2:5].float() * std[2:5] + mean[2:5]
    distance = torch.cdist(xyz, xyz)
    valid_pair = mask[:, :, None] & mask[:, None, :]
    return distance.masked_fill(~valid_pair, 0).amax(dim=(1, 2))


@torch.inference_mode()
def validation_spans(loader, mean, std) -> np.ndarray:
    values = []
    for x, _target, mask, _particle, _point, _valid in loader:
        if len(x):
            values.append(hit_span_m(x, mask, mean, std).numpy())
    return np.concatenate(values)


def validation(model, pred_loader, gt_loader, spans, device) -> dict:
    pred_metrics, pred_rows = evaluate(model, pred_loader, device, return_arrays=True)
    gt_metrics = evaluate(model, gt_loader, device)
    if len(spans) != len(pred_rows["angle"]):
        raise AssertionError("Predicted-hit span and error arrays do not align")
    short = spans < 120.0
    short_error = pred_rows["angle"][short]
    result = {
        "pred_val": pred_metrics,
        "gt_val_clean": gt_metrics,
        "short_pred_val": {
            "count": int(short.sum()),
            "q50_deg": float(np.quantile(short_error, 0.50)),
            "q68_deg": float(np.quantile(short_error, 0.68)),
            "q95_deg": float(np.quantile(short_error, 0.95)),
            "error_ge20_fraction": float(np.mean(short_error >= 20)),
        },
    }
    return result


def selection_score(result: dict) -> float:
    return (
        result["short_pred_val"]["q68_deg"]
        + 0.5 * result["pred_val"]["direction_deg"]["q68"]
        + 0.3 * result["gt_val_clean"]["direction_deg"]["q68"]
        + 0.05 * result["pred_val"]["muatm"]["direction_deg"]["q68"]
        + 0.03 * result["pred_val"]["line_m"]["q68"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--pred-data", type=Path, required=True)
    parser.add_argument("--pred-anchors", type=Path, required=True)
    parser.add_argument("--gt-data", type=Path, required=True)
    parser.add_argument("--gt-anchors", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--short-weight", type=float, default=4.0)
    parser.add_argument("--short-span-m", type=float, default=120.0)
    parser.add_argument("--short-transition-m", type=float, default=15.0)
    parser.add_argument("--gt-weight", type=float, default=0.4)
    parser.add_argument("--point-weight", type=float, default=0.5)
    parser.add_argument("--backbone-lr", type=float, default=5e-6)
    parser.add_argument("--point-lr", type=float, default=1e-5)
    parser.add_argument("--gt-val-stride", type=int, default=6)
    parser.add_argument("--seed", type=int, default=811)
    args = parser.parse_args()
    if (
        min(
            args.steps, args.val_every, args.batch_size, args.short_transition_m, args.gt_val_stride
        )
        <= 0
    ):
        raise ValueError("Counts, transition width, and stride must be positive")
    if args.short_weight < 1 or not 0 <= args.gt_weight <= 1:
        raise ValueError("Invalid short or clean-stream weight")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n"
    )
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
    full_model = build_model(bundle["model_config"])
    full_model.load_state_dict(bundle["state_dict"], strict=True)
    model = full_model.base.to(device)
    mean = np.asarray(bundle["preprocessing"]["input_norm_mean"], dtype=np.float32)
    std = np.asarray(bundle["preprocessing"]["input_norm_std"], dtype=np.float32)
    mean_device = torch.from_numpy(mean).to(device)
    std_device = torch.from_numpy(std).to(device)
    with np.load(args.masks) as masks:
        pred_val_keep = np.asarray(masks["pred_val"], dtype=bool).copy()
        gt_train_keep = np.asarray(masks["gt_train"], dtype=bool).copy()
        gt_val_keep = np.asarray(masks["gt_val"], dtype=bool).copy()
    gt_val_keep &= np.arange(len(gt_val_keep)) % args.gt_val_stride == 0
    pred_val = data_loader(
        args.pred_data,
        args.pred_anchors,
        "val",
        args.batch_size,
        128,
        mean,
        std,
        args.workers,
        args.seed,
        keep=pred_val_keep,
        single_muatm_only=True,
    )
    gt_val = data_loader(
        args.gt_data,
        args.gt_anchors,
        "val",
        args.batch_size,
        128,
        mean,
        std,
        args.workers,
        args.seed + 1,
        keep=gt_val_keep,
    )
    pred_train = infinite_batches(
        data_loader(
            args.pred_data,
            args.pred_anchors,
            "train",
            args.batch_size,
            128,
            mean,
            std,
            args.workers,
            args.seed + 2,
            single_muatm_only=True,
            thinning=True,
            shuffle=True,
        )
    )
    gt_train = infinite_batches(
        data_loader(
            args.gt_data,
            args.gt_anchors,
            "train",
            args.batch_size,
            128,
            mean,
            std,
            args.workers,
            args.seed + 3,
            keep=gt_train_keep,
            shuffle=True,
        )
    )
    spans = validation_spans(pred_val, torch.from_numpy(mean), torch.from_numpy(std))
    print(f"validation short events={int(np.sum(spans < 120))}/{len(spans)}", flush=True)
    initial = validation(model, pred_val, gt_val, spans, device)
    initial.update(step=0, score=selection_score(initial))
    (args.output_dir / "val_step_0.json").write_text(json.dumps(initial, indent=2) + "\n")
    print("validation " + json.dumps(initial, sort_keys=True), flush=True)
    best_score = initial["score"]
    pred_limit = initial["pred_val"]["direction_deg"]["q68"] + 0.05
    gt_limit = initial["gt_val_clean"]["direction_deg"]["q68"] + 0.05
    muatm_limit = initial["pred_val"]["muatm"]["direction_deg"]["q68"] + 0.30
    short_limit = initial["short_pred_val"]["q68_deg"]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {"params": model.point_head.parameters(), "lr": args.point_lr},
        ],
        weight_decay=1e-4,
        fused=device.type == "cuda",
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = {}
        for name, source, weight in (
            ("pred", pred_train, 1.0 - args.gt_weight),
            ("gt", gt_train, args.gt_weight),
        ):
            x, target, mask, _particle, target_point, valid = next_nonempty(source)
            x, target, mask = x.to(device), target.to(device), mask.to(device)
            target_point, valid = target_point.to(device), valid.to(device)
            event_weight = None
            if name == "pred":
                with torch.no_grad():
                    span = hit_span_m(x, mask, mean_device, std_device)
                    emphasis = torch.sigmoid((args.short_span_m - span) / args.short_transition_m)
                    event_weight = 1 + (args.short_weight - 1) * emphasis
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                direction, point = model(x, mask)
                angular = loss_function(
                    direction,
                    target,
                    "mae" if name == "pred" else "trimmed_angular",
                    1.0,
                    event_weight,
                    8.0,
                )
                transverse = point_loss(point, target_point, target, valid, 50.0, "line_huber")
                loss = weight * (angular + args.point_weight * transverse)
            scaler.scale(loss).backward()
            losses[name] = float(loss.detach())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step % 100 == 0:
            print(f"step={step} losses={losses}", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            result = validation(model, pred_val, gt_val, spans, device)
            result.update(step=step, score=selection_score(result))
            result["admissible"] = (
                result["pred_val"]["direction_deg"]["q68"] <= pred_limit
                and result["gt_val_clean"]["direction_deg"]["q68"] <= gt_limit
                and result["pred_val"]["muatm"]["direction_deg"]["q68"] <= muatm_limit
                and result["short_pred_val"]["q68_deg"] < short_limit
            )
            (args.output_dir / f"val_step_{step}.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print("validation " + json.dumps(result, sort_keys=True), flush=True)
            if result["admissible"] and result["score"] < best_score:
                best_score = result["score"]
                torch.save(
                    {"model": model.state_dict(), "args": vars(args), "metrics": result},
                    args.output_dir / "best.pt",
                )
    print(f"best_score={best_score:.6f}", flush=True)


if __name__ == "__main__":
    main()
