#!/usr/bin/env python3
"""Train one shared signal/angle/line model on unfiltered noisy hit clouds."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from joint_raw_dataset import IndexedRawBatchDataset
from joint_raw_model import build_pretrained_joint_model
from torch import nn
from torch.utils.data import DataLoader
from train_angle_signal import angular_errors_degrees, infinite_batches, loss_function
from train_track_anchor_v3 import point_loss


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {f"q{int(q * 100)}": float(np.quantile(values, q)) for q in (0.5, 0.68, 0.95)}


def set_encoder_trainable(model: nn.Module, enabled: bool) -> None:
    for component in (model.first_layer, model.shared, model.cls_head):
        component.requires_grad_(enabled)


def target_offset_from_raw_point(
    true_point_m: torch.Tensor, centroid_m: torch.Tensor, true_direction: torch.Tensor
) -> torch.Tensor:
    delta = true_point_m - centroid_m
    return delta - true_direction * (delta * true_direction).sum(dim=1, keepdim=True)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    angle_errors: list[np.ndarray] = []
    line_errors: list[np.ndarray] = []
    point_valids: list[np.ndarray] = []
    particles: list[np.ndarray] = []
    selected_tp = selected_total = truth_total = 0
    for x, mask, signal, direction, raw_point, valid, particle in loader:
        x, mask = x.to(device), mask.to(device)
        direction, raw_point = direction.to(device), raw_point.to(device)
        valid = valid.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction, offset, center, _logit, probability = model(x, mask)
        prediction, offset, center = prediction.float(), offset.float(), center.float()
        angle_errors.append(angular_errors_degrees(prediction, direction).cpu().numpy())
        target_offset = target_offset_from_raw_point(raw_point, center, direction)
        difference = offset - target_offset
        transverse = difference - direction * (difference * direction).sum(dim=1, keepdim=True)
        line_errors.append(torch.linalg.vector_norm(transverse, dim=1).cpu().numpy())
        point_valids.append(valid.cpu().numpy())
        particles.append(particle.numpy())
        chosen = (probability.float() >= 0.70) & mask
        truth = (signal.to(device) > 0.5) & mask
        selected_tp += int((chosen & truth).sum())
        selected_total += int(chosen.sum())
        truth_total += int(truth.sum())
    angles = np.concatenate(angle_errors)
    lines = np.concatenate(line_errors)
    point_valid = np.concatenate(point_valids)
    particle = np.concatenate(particles)
    result = {
        "count": int(len(angles)),
        "direction_deg": quantiles(angles),
        "line_m": quantiles(lines[point_valid]),
        "point_valid": int(point_valid.sum()),
        "hit_precision_p070": selected_tp / max(selected_total, 1),
        "hit_recall_p070": selected_tp / max(truth_total, 1),
    }
    for code, name in ((0, "muatm"), (1, "nuatm"), (2, "nue2")):
        selected = particle == code
        if selected.any():
            result[name] = {
                "count": int(selected.sum()),
                "direction_deg": quantiles(angles[selected]),
                "line_m": quantiles(lines[selected & point_valid]),
            }
    return result


def validation_score(metrics: dict) -> float:
    return (
        metrics["direction_deg"]["q68"]
        + 0.05 * metrics["line_m"]["q68"]
        + 0.03 * metrics["muatm"]["direction_deg"]["q68"]
    )


def save_checkpoint(
    path: Path, model: nn.Module, optimizer, step: int, args, metrics: dict
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-mc", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--signal-checkpoint", type=Path, required=True)
    parser.add_argument("--model-norm-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-events", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--warm-steps", type=int, default=1000)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--point-weight", type=float, default=0.5)
    parser.add_argument("--hit-weight", type=float, default=0.1)
    parser.add_argument("--muatm-angle-weight", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=421)
    parser.add_argument("--hard-padding-mask", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    if min(args.batch_events, args.max_seq_len, args.steps, args.val_every) < 1:
        raise ValueError(
            "Batch size, sequence length, steps, and validation interval must be positive"
        )
    if min(args.backbone_lr, args.head_lr, args.point_weight, args.hit_weight) < 0:
        raise ValueError("Learning rates and loss weights must be nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n"
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = IndexedRawBatchDataset(
        args.raw_mc,
        args.index,
        "train",
        args.model_norm_source,
        args.batch_events,
        args.max_seq_len,
    )
    val_data = IndexedRawBatchDataset(
        args.raw_mc,
        args.index,
        "val",
        args.model_norm_source,
        args.batch_events,
        args.max_seq_len,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=None,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    source = infinite_batches(train_loader)
    model = build_pretrained_joint_model(
        args.signal_checkpoint,
        args.model_norm_source,
        legacy_float_mask=not args.hard_padding_mask,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.first_layer.parameters())
                + list(model.shared.parameters())
                + list(model.cls_head.parameters()),
                "lr": args.backbone_lr,
            },
            {
                "params": list(model.direction_head.parameters())
                + list(model.point_head.parameters()),
                "lr": args.head_lr,
            },
        ],
        weight_decay=1e-4,
        fused=device.type == "cuda",
    )
    initial_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if bool(checkpoint["args"].get("hard_padding_mask", False)) != args.hard_padding_mask:
            raise ValueError("Resume checkpoint padding-mask mode differs from current run")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer.param_groups[0]["lr"] = args.backbone_lr
        optimizer.param_groups[1]["lr"] = args.head_lr
        initial_step = int(checkpoint["step"])
    set_encoder_trainable(model, initial_step >= args.warm_steps)
    best_score = float("inf")
    if initial_step == 0:
        initial = evaluate(model, val_loader, device)
        initial["step"] = 0
        initial["score"] = validation_score(initial)
        (args.output_dir / "val_step_0.json").write_text(json.dumps(initial, indent=2) + "\n")
        print("validation " + json.dumps(initial, sort_keys=True), flush=True)
    for step in range(initial_step + 1, args.steps + 1):
        if step == args.warm_steps + 1:
            set_encoder_trainable(model, True)
            print(f"unfroze_shared_encoder_at_step={step}", flush=True)
        model.train()
        if step <= args.warm_steps:
            model.first_layer.eval()
            model.shared.eval()
            model.cls_head.eval()
        x, mask, signal, target_direction, raw_point, valid, particle = next(source)
        x, mask, signal = x.to(device), mask.to(device), signal.to(device)
        target_direction = target_direction.to(device)
        raw_point, valid = raw_point.to(device), valid.to(device)
        particle = particle.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            direction, offset, center, hit_logit, _probability = model(x, mask)
            weights = torch.where(
                particle == 0,
                torch.full_like(particle, args.muatm_angle_weight, dtype=torch.float32),
                torch.ones_like(particle, dtype=torch.float32),
            )
            angle_loss = loss_function(direction, target_direction, "mae", 1.0, weights)
            target_offset = target_offset_from_raw_point(
                raw_point, center.float(), target_direction
            )
            line_loss = point_loss(
                offset.float(),
                target_offset,
                target_direction,
                valid,
                50.0,
                "line_huber",
            )
            hit_loss = F.binary_cross_entropy_with_logits(hit_logit[mask], signal[mask])
            total = angle_loss + args.point_weight * line_loss
            if step > args.warm_steps:
                total = total + args.hit_weight * hit_loss
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            print(
                f"step={step} angle={float(angle_loss):.5f} line={float(line_loss):.5f} "
                f"hit={float(hit_loss):.5f}",
                flush=True,
            )
        if step % args.val_every == 0 or step == args.steps:
            result = evaluate(model, val_loader, device)
            result["step"] = step
            result["score"] = validation_score(result)
            (args.output_dir / f"val_step_{step}.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            save_checkpoint(args.output_dir / "last.pt", model, optimizer, step, args, result)
            if result["score"] < best_score:
                best_score = result["score"]
                save_checkpoint(args.output_dir / "best.pt", model, optimizer, step, args, result)
            print("validation " + json.dumps(result, sort_keys=True), flush=True)
    print(f"best_score={best_score}", flush=True)


if __name__ == "__main__":
    main()
