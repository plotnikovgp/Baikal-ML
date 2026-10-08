#!/usr/bin/env python3
"""Fine-tune one angle/point model on predicted hits and nue2 GT hits.

Optional single-muatm filtering removes events with multiple simulated muons
from predicted-hit training and validation. The model is initialized from the
nue2 angle/point specialist. Cross-dataset masks keep evaluation independent
of both that specialist and the older all-particle predicted-hit model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from train_angle_signal import (
    SetTransformerDirection,
    infinite_batches,
    loss_function,
    make_loader,
    seed_everything,
)
from train_track_anchor_v3 import BatchedTrackDataset, DirectionTrackAnchor, point_loss
from train_track_uncertainty_v2 import residuals


class AdaptedTrackDataset(BatchedTrackDataset):
    def __init__(
        self,
        *args,
        keep: np.ndarray | None,
        expected_mean,
        expected_std,
        single_muatm_only: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.keep = keep
        self.single_muatm_only = single_muatm_only
        if keep is not None and len(keep) < self.n_events:
            raise ValueError(f"Selection mask too short for {self.split}")
        self.source_mean = torch.tensor(self.target_mean, dtype=torch.float32)
        self.source_std = torch.tensor(self.target_std, dtype=torch.float32)
        self.expected_mean = torch.tensor(expected_mean, dtype=torch.float32)
        self.expected_std = torch.tensor(expected_std, dtype=torch.float32)

    def __getitem__(self, batch_index: int):
        x, direction, mask, particle, point, valid = super().__getitem__(batch_index)
        x = (x * self.source_std + self.source_mean - self.expected_mean) / self.expected_std
        x = x * mask.unsqueeze(-1)
        selection = torch.ones(len(x), dtype=torch.bool)
        if self.keep is not None:
            begin = batch_index * self.events_per_batch
            selection &= torch.from_numpy(self.keep[begin : begin + len(x)].copy())
        if self.single_muatm_only:
            selection &= (particle != 0) | valid
        if not torch.all(selection):
            x, direction, mask, particle, point, valid = (
                value[selection] for value in (x, direction, mask, particle, point, valid)
            )
        return x, direction, mask, particle, point, valid


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "q50": float(np.quantile(values, 0.5)),
        "q68": float(np.quantile(values, 0.68)),
        "q95": float(np.quantile(values, 0.95)),
    }


def next_nonempty(source):
    while True:
        batch = next(source)
        if len(batch[0]):
            return batch


def validation_score(result: dict, pred_weight: float, muatm_weight: float) -> float:
    pred = result["pred_val"]
    gt = result["gt_val_clean"]
    return (
        gt["direction_deg"]["q68"]
        + 0.03 * gt["line_m"]["q68"]
        + pred_weight * pred["direction_deg"]["q68"]
        + muatm_weight * pred["muatm"]["direction_deg"]["q68"]
    )


@torch.inference_mode()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    chunks = {name: [] for name in ("angle", "line", "valid", "particle")}
    for x, target_dir, mask, particle, target_point, valid in loader:
        if not len(x):
            continue
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        target_dir = target_dir.to(device, non_blocking=True)
        target_point = target_point.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            direction, point = model(x, mask)
        angle, line, _ = residuals(direction.float(), point.float(), target_dir, target_point)
        chunks["angle"].append(angle.cpu().numpy())
        chunks["line"].append(line.cpu().numpy())
        chunks["valid"].append(valid.numpy())
        chunks["particle"].append(particle.numpy())
    pred = {key: np.concatenate(value) for key, value in chunks.items()}
    result = {
        "count": int(len(pred["angle"])),
        "track_count": int(pred["valid"].sum()),
        "direction_deg": quantiles(pred["angle"]),
        "line_m": quantiles(pred["line"][pred["valid"]]),
    }
    for code, name in ((0, "muatm"), (1, "nuatm"), (2, "nue2")):
        selected = pred["particle"] == code
        if selected.any():
            track = selected & pred["valid"]
            result[name] = {
                "count": int(selected.sum()),
                "track_count": int(track.sum()),
                "direction_deg": quantiles(pred["angle"][selected]),
            }
            if track.any():
                result[name]["line_m"] = quantiles(pred["line"][track])
    return result


def model_from_checkpoints(angle_path: Path, track_path: Path, scale: float, device):
    angle = torch.load(angle_path, map_location="cpu", weights_only=False)
    config = angle["args"]
    backbone = SetTransformerDirection(
        config["hidden"],
        config["layers"],
        config["heads"],
        config["ff_mult"],
        config["dropout"],
        config["feature_mode"],
    )
    model = DirectionTrackAnchor(backbone, config["hidden"], config["dropout"], scale)
    track = torch.load(track_path, map_location="cpu", weights_only=False)
    model.load_state_dict(track["model"], strict=True)
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-data", type=Path, required=True)
    parser.add_argument("--norm-data", type=Path, default=None)
    parser.add_argument("--pred-anchors", type=Path, required=True)
    parser.add_argument("--gt-data", type=Path, required=True)
    parser.add_argument("--gt-anchors", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--angle-checkpoint", type=Path, required=True)
    parser.add_argument("--track-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--gt-weight", type=float, default=0.5)
    parser.add_argument("--point-weight", type=float, default=0.5)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--point-lr", type=float, default=1e-4)
    parser.add_argument("--direction-head-only", action="store_true")
    parser.add_argument("--teacher-weight", type=float, default=0.0)
    parser.add_argument("--muatm-teacher-angle", type=Path, default=None)
    parser.add_argument("--muatm-teacher-track", type=Path, default=None)
    parser.add_argument("--max-pred-q68", type=float, default=None)
    parser.add_argument("--max-gt-q68", type=float, default=None)
    parser.add_argument("--pred-score-weight", type=float, default=0.0)
    parser.add_argument("--muatm-score-weight", type=float, default=0.0)
    parser.add_argument("--single-muatm-only", action="store_true")
    parser.add_argument("--anchor-scale-m", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-split", choices=("val", "test"), default="val")
    parser.add_argument("--load-model", type=Path, default=None)
    args = parser.parse_args()
    if not 0 <= args.gt_weight <= 1:
        raise ValueError("gt-weight must be between 0 and 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n"
    )
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_from_checkpoints(
        args.angle_checkpoint, args.track_checkpoint, args.anchor_scale_m, device
    )
    if args.load_model is not None:
        checkpoint = torch.load(args.load_model, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
    teacher = None
    muatm_teacher = None
    if args.teacher_weight > 0 and not args.eval_only:
        teacher = model_from_checkpoints(
            args.angle_checkpoint, args.track_checkpoint, args.anchor_scale_m, device
        )
        teacher.eval().requires_grad_(False)
        if (args.muatm_teacher_angle is None) != (args.muatm_teacher_track is None):
            raise ValueError("Both muatm teacher checkpoints must be specified together")
        if args.muatm_teacher_angle is not None:
            muatm_teacher = model_from_checkpoints(
                args.muatm_teacher_angle,
                args.muatm_teacher_track,
                args.anchor_scale_m,
                device,
            )
            muatm_teacher.eval().requires_grad_(False)
    with h5py.File(args.norm_data or args.pred_data, "r") as h5:
        expected_mean = np.asarray(h5["norm_param/mean"], dtype=np.float32)
        expected_std = np.asarray(h5["norm_param/std"], dtype=np.float32)
    with h5py.File(args.pred_data, "r") as h5:
        pred_mean = np.asarray(h5["norm_param/mean"], dtype=np.float32)
        pred_std = np.asarray(h5["norm_param/std"], dtype=np.float32)
    student_mean = torch.as_tensor(expected_mean, device=device)
    student_std = torch.as_tensor(expected_std, device=device)
    teacher_mean = torch.as_tensor(pred_mean, device=device)
    teacher_std = torch.as_tensor(pred_std, device=device)
    with np.load(args.masks) as mask_file:
        gt_masks = {name: mask_file[name].copy() for name in ("gt_train", "gt_val", "gt_test")}
        pred_masks = {
            split: mask_file[f"pred_{split}"].copy() if f"pred_{split}" in mask_file else None
            for split in ("val", "test")
        }
    # Spread validation over the entire nue2 GT file, rather than taking the
    # first contiguous block of its event ordering.
    gt_val_small = gt_masks["gt_val"] & (np.arange(len(gt_masks["gt_val"])) % 6 == 0)

    def dataset(path, anchors, split, keep=None, single_muatm_only=False):
        return AdaptedTrackDataset(
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
            single_muatm_only=single_muatm_only,
        )

    if args.eval_only:
        split = args.eval_split
        selection = gt_val_small if split == "val" else gt_masks["gt_test"]
        pred_eval = make_loader(
            dataset(
                args.pred_data,
                args.pred_anchors,
                split,
                keep=pred_masks[split],
                single_muatm_only=args.single_muatm_only,
            ),
            args.workers,
            False,
            args.seed,
        )
        gt_eval = make_loader(
            dataset(args.gt_data, args.gt_anchors, split, selection), args.workers, False, args.seed
        )
        result = {
            f"pred_{split}": evaluate(model, pred_eval, device),
            f"gt_{split}_clean": evaluate(model, gt_eval, device),
        }
        if args.single_muatm_only or pred_masks[split] is not None:
            pred_full = make_loader(
                dataset(args.pred_data, args.pred_anchors, split), args.workers, False, args.seed
            )
            result[f"pred_{split}_full_diagnostic"] = evaluate(model, pred_full, device)
        (args.output_dir / "eval.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        return

    pred_val = make_loader(
        dataset(
            args.pred_data,
            args.pred_anchors,
            "val",
            keep=pred_masks["val"],
            single_muatm_only=args.single_muatm_only,
        ),
        args.workers,
        False,
        args.seed,
    )
    gt_val = make_loader(
        dataset(args.gt_data, args.gt_anchors, "val", gt_val_small), args.workers, False, args.seed
    )

    pred_train = infinite_batches(
        make_loader(
            dataset(
                args.pred_data, args.pred_anchors, "train", single_muatm_only=args.single_muatm_only
            ),
            args.workers,
            True,
            args.seed,
        )
    )
    gt_train = infinite_batches(
        make_loader(
            dataset(args.gt_data, args.gt_anchors, "train", gt_masks["gt_train"]),
            args.workers,
            True,
            args.seed + 1,
        )
    )
    if args.direction_head_only:
        model.backbone.requires_grad_(False)
        model.backbone.head.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.backbone.head.parameters()
                if args.direction_head_only
                else model.backbone.parameters(),
                "lr": args.backbone_lr,
            },
            {"params": model.point_head.parameters(), "lr": args.point_lr},
        ],
        weight_decay=1e-4,
        fused=device.type == "cuda",
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    initial = {
        "pred_val": evaluate(model, pred_val, device),
        "gt_val_clean": evaluate(model, gt_val, device),
    }
    initial["step"] = 0
    initial["score"] = validation_score(initial, args.pred_score_weight, args.muatm_score_weight)
    (args.output_dir / "val_step_0.json").write_text(json.dumps(initial, indent=2) + "\n")
    print("validation " + json.dumps(initial, sort_keys=True), flush=True)
    baseline = initial["pred_val"]["direction_deg"]["q68"]
    limit = args.max_pred_q68 if args.max_pred_q68 is not None else 1.05 * baseline
    initial_gt_q68 = initial["gt_val_clean"]["direction_deg"]["q68"]
    initial_admissible = baseline <= limit and (
        args.max_gt_q68 is None or initial_gt_q68 <= args.max_gt_q68
    )
    best_score = initial["score"] if initial_admissible else float("inf")
    for step in range(1, args.steps + 1):
        model.train()
        if args.direction_head_only:
            model.backbone.embed.eval()
            model.backbone.encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        losses = {}
        for name, source, weight in (
            ("pred", pred_train, 1 - args.gt_weight),
            ("gt", gt_train, args.gt_weight),
        ):
            x, target_dir, mask, particle, target_point, valid = next_nonempty(source)
            x = x.to(device, non_blocking=True)
            target_dir = target_dir.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            target_point = target_point.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                direction, point = model(x, mask)
                angular = loss_function(
                    direction,
                    target_dir,
                    "mae" if name == "pred" else "trimmed_angular",
                    1.0,
                    None,
                    8.0,
                )
                transverse = point_loss(
                    point, target_point, target_dir, valid, args.anchor_scale_m, "line_huber"
                )
                loss = weight * (angular + args.point_weight * transverse)
                if name == "pred" and teacher is not None:
                    with torch.no_grad():
                        teacher_dir, teacher_point = teacher(x, mask)
                        if muatm_teacher is not None and torch.any(particle == 0):
                            teacher_x = (
                                x * student_std + student_mean - teacher_mean
                            ) / teacher_std
                            teacher_x = teacher_x * mask.unsqueeze(-1)
                            mu_dir, mu_point = muatm_teacher(teacher_x, mask)
                            use_mu = (particle.to(device) == 0).unsqueeze(-1)
                            teacher_dir = torch.where(use_mu, mu_dir, teacher_dir)
                            teacher_point = torch.where(use_mu, mu_point, teacher_point)
                    direction_distill = F.l1_loss(direction.float(), teacher_dir.float())
                    point_distill = F.smooth_l1_loss(
                        (point.float() - teacher_point.float()) / args.anchor_scale_m,
                        torch.zeros_like(point.float()),
                        beta=0.1,
                    )
                    loss = loss + weight * args.teacher_weight * (
                        direction_distill + args.point_weight * point_distill
                    )
            scaler.scale(loss).backward()
            losses[name] = float(loss.detach())
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step % 50 == 0:
            print(f"step={step} losses={losses}", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            result = {
                "pred_val": evaluate(model, pred_val, device),
                "gt_val_clean": evaluate(model, gt_val, device),
            }
            pred_q68 = result["pred_val"]["direction_deg"]["q68"]
            gt_q68 = result["gt_val_clean"]["direction_deg"]["q68"]
            # Primary: clean nue2 direction. Reject candidates that damage
            # all-particle direction by more than 5% on its own validation set.
            admissible = pred_q68 <= limit and (
                args.max_gt_q68 is None or gt_q68 <= args.max_gt_q68
            )
            score = validation_score(result, args.pred_score_weight, args.muatm_score_weight)
            result["step"] = step
            result["score"] = score
            result["admissible"] = admissible
            print("validation " + json.dumps(result, sort_keys=True), flush=True)
            (args.output_dir / f"val_step_{step}.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            if admissible and score < best_score:
                best_score = score
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": vars(args),
                        "metrics": result,
                        "step": step,
                    },
                    args.output_dir / "best.pt",
                )
    print(f"best_score={best_score}", flush=True)


if __name__ == "__main__":
    main()
