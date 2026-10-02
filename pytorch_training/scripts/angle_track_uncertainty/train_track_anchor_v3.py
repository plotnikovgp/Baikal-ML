#!/usr/bin/env python3
"""Multitask direction and track-anchor training for single-particle nue2 events."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from train_angle_signal import (
    BatchedAngleDataset,
    SetTransformerDirection,
    angular_errors_degrees,
    infinite_batches,
    loss_function,
    make_loader,
    seed_everything,
)


class BatchedTrackDataset(BatchedAngleDataset):
    def __init__(self, *args, anchor_path: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.anchor_path = anchor_path
        self._anchor_file = None
        with h5py.File(anchor_path, "r") as anchors:
            anchor_count = len(anchors[f"{self.split}/valid"])
            if anchor_count < self.n_events:
                raise ValueError(
                    f"Anchor file has {anchor_count} {self.split} rows, expected {self.n_events}"
                )
            with h5py.File(self.path, "r") as data:
                data_ids = data[f"{self.split}/ev_ids/data"]
                anchor_ids = anchors[f"{self.split}/event_ids"]
                for index in (0, self.n_events // 2, self.n_events - 1):
                    if data_ids[index] != anchor_ids[index]:
                        raise ValueError(f"Event-ID mismatch at {self.split}[{index}]")

    @property
    def anchor_h5(self) -> h5py.File:
        if self._anchor_file is None:
            self._anchor_file = h5py.File(self.anchor_path, "r", swmr=True)
        return self._anchor_file

    def __getstate__(self):
        state = super().__getstate__()
        state["_anchor_file"] = None
        return state

    def __getitem__(self, batch_index: int):
        x, direction, mask, particle = super().__getitem__(batch_index)
        event_start = batch_index * self.events_per_batch
        event_end = min(event_start + self.events_per_batch, self.n_events)
        offsets = np.asarray(
            self.anchor_h5[f"{self.split}/anchor_offset_m"][event_start:event_end],
            dtype=np.float32,
        )
        valid = np.asarray(self.anchor_h5[f"{self.split}/valid"][event_start:event_end], dtype=bool)
        offsets[~valid] = 0.0
        return x, direction, mask, particle, torch.from_numpy(offsets), torch.from_numpy(valid)


class DirectionTrackAnchor(nn.Module):
    def __init__(
        self, backbone: SetTransformerDirection, hidden: int, dropout: float, scale_m: float
    ):
        super().__init__()
        self.backbone = backbone
        self.scale_m = scale_m
        self.point_head = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.point_head[-1].weight)
        nn.init.zeros_(self.point_head[-1].bias)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        direction, pooled = self.backbone(x, mask, return_features=True)
        raw_offset = self.point_head(pooled) * self.scale_m
        projection_direction = direction.detach()
        offset = raw_offset - projection_direction * torch.sum(
            raw_offset * projection_direction, dim=1, keepdim=True
        )
        return direction, offset


def quantiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_q50": float(np.quantile(values, 0.50)),
        f"{prefix}_q68": float(np.quantile(values, 0.68)),
        f"{prefix}_q90": float(np.quantile(values, 0.90)),
    }


def point_loss(
    prediction: Tensor,
    target: Tensor,
    target_direction: Tensor,
    valid: Tensor,
    scale_m: float,
    kind: str,
) -> Tensor:
    if not torch.any(valid):
        return prediction.sum() * 0.0
    residual = (prediction[valid] - target[valid]) / scale_m
    if kind == "line_huber":
        direction = target_direction[valid]
        transverse = residual - direction * torch.sum(residual * direction, dim=1, keepdim=True)
        distance = torch.linalg.vector_norm(transverse, dim=1)
        return F.huber_loss(distance, torch.zeros_like(distance), delta=0.2)
    if kind == "euclidean_huber":
        distance = torch.linalg.vector_norm(residual, dim=1)
        return F.huber_loss(distance, torch.zeros_like(distance), delta=0.5)
    if kind == "log_cosh":
        return torch.log(torch.cosh(residual.clamp(-10, 10))).mean()
    return F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=0.2)


@torch.inference_mode()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    direction_errors = []
    anchor_errors = []
    line_errors = []
    centroid_line_errors = []
    valid_count = 0
    event_count = 0
    for x, target_direction, mask, _particle, target_offset, valid in loader:
        x = x.to(device, non_blocking=True)
        target_direction = target_direction.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        target_offset = target_offset.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            prediction_direction, prediction_offset = model(x, mask)
        prediction_direction = prediction_direction.float()
        prediction_offset = prediction_offset.float()
        direction_errors.append(
            angular_errors_degrees(prediction_direction, target_direction).cpu().numpy()
        )
        event_count += len(x)
        if torch.any(valid):
            difference = prediction_offset[valid] - target_offset[valid]
            truth_direction = target_direction[valid]
            transverse = difference - truth_direction * torch.sum(
                difference * truth_direction, dim=1, keepdim=True
            )
            anchor_errors.append(torch.linalg.vector_norm(difference, dim=1).cpu().numpy())
            line_errors.append(torch.linalg.vector_norm(transverse, dim=1).cpu().numpy())
            centroid_line_errors.append(
                torch.linalg.vector_norm(target_offset[valid], dim=1).cpu().numpy()
            )
            valid_count += int(valid.sum())
    angular = np.concatenate(direction_errors)
    anchor = np.concatenate(anchor_errors)
    line = np.concatenate(line_errors)
    centroid = np.concatenate(centroid_line_errors)
    return {
        "count": event_count,
        "anchor_count": valid_count,
        **quantiles(angular, "direction_deg"),
        **quantiles(anchor, "anchor_error_m"),
        **quantiles(line, "line_distance_m"),
        **quantiles(centroid, "centroid_baseline_m"),
    }


def append_csv(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, step, args, metrics) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--init-track-checkpoint", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-train-events", type=int, default=None)
    parser.add_argument("--max-val-events", type=int, default=50_000)
    parser.add_argument("--max-test-events", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--freeze-backbone-steps", type=int, default=500)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.01)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--direction-loss", default="trimmed_angular")
    parser.add_argument("--direction-cap-deg", type=float, default=8.0)
    parser.add_argument(
        "--point-loss",
        choices=("smooth_l1", "euclidean_huber", "line_huber", "log_cosh"),
        default="smooth_l1",
    )
    parser.add_argument("--point-weight", type=float, default=0.05)
    parser.add_argument("--anchor-scale-m", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    initial = torch.load(args.init_checkpoint, map_location="cpu")
    backbone_args = argparse.Namespace(**initial["args"])
    if backbone_args.model != "transformer":
        raise ValueError("Track-anchor harness currently expects the winning transformer")
    backbone = SetTransformerDirection(
        backbone_args.hidden,
        backbone_args.layers,
        backbone_args.heads,
        backbone_args.ff_mult,
        backbone_args.dropout,
        backbone_args.feature_mode,
    )
    backbone.load_state_dict(initial["model"])
    model = DirectionTrackAnchor(
        backbone, backbone_args.hidden, backbone_args.dropout, args.anchor_scale_m
    ).to(device)
    if args.init_track_checkpoint:
        track_initial = torch.load(args.init_track_checkpoint, map_location=device)
        model.load_state_dict(track_initial["model"])
        print(f"initialized multitask weights from {args.init_track_checkpoint}", flush=True)
    print(f"device={device}; parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)

    def dataset(split: str, max_events: int | None):
        return BatchedTrackDataset(
            args.data,
            split,
            args.batch_size,
            args.max_seq_len,
            bool(getattr(backbone_args, "use_signal_norm", False)),
            max_events,
            bool(getattr(backbone_args, "center_time", False)),
            anchor_path=args.anchors,
        )

    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--eval-only requires --checkpoint")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        test = dataset("test", args.max_test_events)
        metrics = evaluate(model, make_loader(test, args.workers, False, args.seed), device)
        (output_dir / "test_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
        return

    train = dataset("train", args.max_train_events)
    val = dataset("val", args.max_val_events)
    train_batches = infinite_batches(make_loader(train, args.workers, True, args.seed))
    val_loader = make_loader(val, args.workers, False, args.seed)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr * args.backbone_lr_scale},
            {"params": model.point_head.parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return max((step + 1) / max(args.warmup_steps, 1), args.min_lr / args.lr)
        progress = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return args.min_lr / args.lr + (1.0 - args.min_lr / args.lr) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    backbone_frozen = args.freeze_backbone_steps > 0
    model.backbone.requires_grad_(not backbone_frozen)
    best_line_q68 = float("inf")
    best_direction_q68 = float("inf")
    baseline_direction_q68 = None
    metrics_csv = output_dir / "metrics.csv"
    running = {"total": 0.0, "direction": 0.0, "point": 0.0, "count": 0}
    last_time = time.time()

    for step in range(1, args.max_steps + 1):
        if backbone_frozen and step > args.freeze_backbone_steps:
            model.backbone.requires_grad_(True)
            backbone_frozen = False
            print(f"unfroze backbone at step={step}", flush=True)
        model.train()
        x, target_direction, mask, _particle, target_offset, valid = next(train_batches)
        x = x.to(device, non_blocking=True)
        target_direction = target_direction.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        target_offset = target_offset.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            prediction_direction, prediction_offset = model(x, mask)
            direction_objective = loss_function(
                prediction_direction,
                target_direction,
                args.direction_loss,
                1.0,
                None,
                args.direction_cap_deg,
            )
            point_objective = point_loss(
                prediction_offset,
                target_offset,
                target_direction,
                valid,
                args.anchor_scale_m,
                args.point_loss,
            )
            total_loss = direction_objective + args.point_weight * point_objective
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running["total"] += float(total_loss.detach())
        running["direction"] += float(direction_objective.detach())
        running["point"] += float(point_objective.detach())
        running["count"] += 1

        if step % args.log_every == 0:
            elapsed = time.time() - last_time
            print(
                f"step={step} total={running['total'] / running['count']:.6f} "
                f"direction={running['direction'] / running['count']:.6f} "
                f"point={running['point'] / running['count']:.6f} "
                f"steps/s={running['count'] / elapsed:.2f}",
                flush=True,
            )
            running = {"total": 0.0, "direction": 0.0, "point": 0.0, "count": 0}
            last_time = time.time()

        if step % args.val_every == 0 or step == args.max_steps:
            metrics = evaluate(model, val_loader, device)
            metrics["step"] = step
            metrics["backbone_frozen"] = backbone_frozen
            metrics["head_lr"] = optimizer.param_groups[1]["lr"]
            metrics["backbone_lr"] = optimizer.param_groups[0]["lr"]
            if baseline_direction_q68 is None:
                baseline_direction_q68 = metrics["direction_deg_q68"]
            print("validation " + json.dumps(metrics, sort_keys=True), flush=True)
            append_csv(metrics_csv, metrics)
            direction_ok = metrics["direction_deg_q68"] <= baseline_direction_q68 * 1.01
            if direction_ok and metrics["line_distance_m_q68"] < best_line_q68:
                best_line_q68 = metrics["line_distance_m_q68"]
                save_checkpoint(
                    output_dir / "best_line_q68.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    step,
                    args,
                    metrics,
                )
            if metrics["direction_deg_q68"] < best_direction_q68:
                best_direction_q68 = metrics["direction_deg_q68"]
                save_checkpoint(
                    output_dir / "best_direction_q68.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    step,
                    args,
                    metrics,
                )
            save_checkpoint(
                output_dir / "last.pt", model, optimizer, scheduler, scaler, step, args, metrics
            )


if __name__ == "__main__":
    main()
