#!/usr/bin/env python3
"""Train or fine-tune one physics-residual direction/line model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from bundle_universal_model import build_model
from photon_residual_upgrade import PhotonResidualUpgrade
from train_angle_signal import (
    infinite_batches,
    loss_function,
    make_loader,
    seed_everything,
)
from train_mixed_universal import (
    AdaptedTrackDataset,
    evaluate,
    next_nonempty,
    validation_score,
)
from train_track_anchor_v3 import point_loss


def data_loader(
    path: Path,
    anchors: Path,
    split: str,
    batch_size: int,
    max_seq_len: int,
    mean: np.ndarray,
    std: np.ndarray,
    workers: int,
    seed: int,
    *,
    keep: np.ndarray | None = None,
    single_muatm_only: bool = False,
    thinning: bool = False,
    paired_threshold: float = 0.0,
    shuffle: bool = False,
):
    data = AdaptedTrackDataset(
        str(path),
        split,
        batch_size,
        max_seq_len,
        False,
        None,
        False,
        anchor_path=str(anchors),
        keep=keep,
        expected_mean=mean,
        expected_std=std,
        single_muatm_only=single_muatm_only,
        pred_thinning_prob=1.0 if paired_threshold else (0.5 if thinning else 0.0),
        pred_threshold_min=paired_threshold if paired_threshold else 0.70,
        pred_threshold_max=paired_threshold if paired_threshold else 0.90,
        return_unthinned=bool(paired_threshold),
    )
    return make_loader(data, workers, shuffle, seed)


def optimizer_groups(
    model: PhotonResidualUpgrade,
    *,
    hit_learning_rate: float,
    base_learning_rate: float,
    train_base_heads: bool,
) -> tuple[list[dict], list[torch.nn.Parameter]]:
    """Materialize parameter lists so clipping cannot exhaust optimizer inputs."""
    groups = [{"params": list(model.hit_head.parameters()), "lr": hit_learning_rate}]
    if train_base_heads:
        model.base.backbone.head.requires_grad_(True)
        model.base.point_head.requires_grad_(True)
        groups.append(
            {
                "params": list(model.base.backbone.head.parameters())
                + list(model.base.point_head.parameters()),
                "lr": base_learning_rate,
            }
        )
    trainable = [parameter for group in groups for parameter in group["params"]]
    if not trainable or not all(parameter.requires_grad for parameter in trainable):
        raise RuntimeError("Optimizer has no valid trainable parameters")
    return groups, trainable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="Resume a trusted local photon-residual checkpoint instead of zero-init",
    )
    parser.add_argument(
        "--train-base-heads",
        action="store_true",
        help="Also train the direction and point heads; keep the encoder frozen",
    )
    parser.add_argument("--base-learning-rate", type=float, default=1e-5)
    parser.add_argument("--pred-data", type=Path, required=True)
    parser.add_argument("--pred-anchors", type=Path, required=True)
    parser.add_argument("--gt-data", type=Path, required=True)
    parser.add_argument("--gt-anchors", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gt-weight", type=float, default=0.4)
    parser.add_argument("--point-weight", type=float, default=0.5)
    parser.add_argument(
        "--paired-threshold",
        type=float,
        default=0.0,
        help="Train simultaneously on original p>=0.70 and a fixed higher-p view",
    )
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--gt-val-stride", type=int, default=6)
    parser.add_argument("--seed", type=int, default=743)
    args = parser.parse_args()
    if (
        min(
            args.steps,
            args.val_every,
            args.batch_size,
            args.max_seq_len,
            args.hidden,
            args.gt_val_stride,
        )
        < 1
    ):
        raise ValueError("Counts must be positive")
    if (
        not 0 <= args.gt_weight <= 1
        or args.learning_rate <= 0
        or args.base_learning_rate <= 0
        or args.point_weight < 0
        or args.consistency_weight < 0
        or (args.paired_threshold and not 0.70 <= args.paired_threshold <= 1.0)
        or (args.consistency_weight and not args.paired_threshold)
    ):
        raise ValueError("Invalid loss weight or learning rate")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n"
    )
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
    full_model = build_model(bundle["model_config"])
    full_model.load_state_dict(bundle["state_dict"], strict=True)
    mean = np.asarray(bundle["preprocessing"]["input_norm_mean"], dtype=np.float32)
    std = np.asarray(bundle["preprocessing"]["input_norm_std"], dtype=np.float32)
    model = PhotonResidualUpgrade(
        full_model.base,
        torch.from_numpy(mean),
        torch.from_numpy(std),
        hidden=args.hidden,
    ).to(device)
    if args.initial_checkpoint is not None:
        saved = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"], strict=True)
    parameter_groups, trainable = optimizer_groups(
        model,
        hit_learning_rate=args.learning_rate,
        base_learning_rate=args.base_learning_rate,
        train_base_heads=args.train_base_heads,
    )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=1e-4,
        fused=device.type == "cuda",
    )
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
        args.max_seq_len,
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
        args.max_seq_len,
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
            args.max_seq_len,
            mean,
            std,
            args.workers,
            args.seed + 2,
            single_muatm_only=True,
            thinning=not bool(args.paired_threshold),
            paired_threshold=args.paired_threshold,
            shuffle=True,
        )
    )
    gt_train = infinite_batches(
        data_loader(
            args.gt_data,
            args.gt_anchors,
            "train",
            args.batch_size,
            args.max_seq_len,
            mean,
            std,
            args.workers,
            args.seed + 3,
            keep=gt_train_keep,
            shuffle=True,
        )
    )
    initial = {
        "pred_val": evaluate(model, pred_val, device),
        "gt_val_clean": evaluate(model, gt_val, device),
    }
    initial["step"] = 0
    initial["score"] = validation_score(initial, 0.5, 0.03)
    (args.output_dir / "val_step_0.json").write_text(json.dumps(initial, indent=2) + "\n")
    print("validation " + json.dumps(initial, sort_keys=True), flush=True)
    best_score = initial["score"]
    pred_limit = initial["pred_val"]["direction_deg"]["q68"]
    gt_limit = initial["gt_val_clean"]["direction_deg"]["q68"] + 0.02
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = {}
        for name, source, weight in (
            ("pred", pred_train, 1.0 - args.gt_weight),
            ("gt", gt_train, args.gt_weight),
        ):
            batch = next_nonempty(source)
            x, target, mask, _particle, target_point, valid = batch[:6]
            x, target, mask = x.to(device), target.to(device), mask.to(device)
            target_point, valid = target_point.to(device), valid.to(device)
            if name == "pred" and args.paired_threshold:
                original_x, original_mask, original_point = batch[6:9]
                original_x = original_x.to(device)
                original_mask = original_mask.to(device)
                original_point = original_point.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                direction, point = model(x, mask)
                angular = loss_function(
                    direction,
                    target,
                    "mae" if name == "pred" else "trimmed_angular",
                    1.0,
                    None,
                    8.0,
                )
                transverse = point_loss(
                    point,
                    target_point,
                    target,
                    valid,
                    50.0,
                    "line_huber",
                )
                if name == "pred" and args.paired_threshold:
                    original_direction, original_prediction_point = model(original_x, original_mask)
                    original_angular = loss_function(
                        original_direction, target, "mae", 1.0, None, 8.0
                    )
                    original_transverse = point_loss(
                        original_prediction_point,
                        original_point,
                        target,
                        valid,
                        50.0,
                        "line_huber",
                    )
                    angular = 0.5 * (angular + original_angular)
                    transverse = 0.5 * (transverse + original_transverse)
                    consistency = loss_function(
                        direction, original_direction, "mae", 1.0, None, 8.0
                    )
                else:
                    consistency = angular.new_zeros(())
                loss = weight * (
                    angular + args.point_weight * transverse + args.consistency_weight * consistency
                )
            loss.backward()
            losses[name] = float(loss.detach())
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step % 100 == 0:
            print(f"step={step} losses={losses}", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            result = {
                "pred_val": evaluate(model, pred_val, device),
                "gt_val_clean": evaluate(model, gt_val, device),
            }
            result["step"] = step
            result["score"] = validation_score(result, 0.5, 0.03)
            result["admissible"] = (
                result["pred_val"]["direction_deg"]["q68"] <= pred_limit
                and result["gt_val_clean"]["direction_deg"]["q68"] <= gt_limit
            )
            (args.output_dir / f"val_step_{step}.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print("validation " + json.dumps(result, sort_keys=True), flush=True)
            if result["admissible"] and result["score"] < best_score:
                best_score = result["score"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": vars(args),
                        "metrics": result,
                    },
                    args.output_dir / "best.pt",
                )
    print(f"best_score={best_score:.6f}", flush=True)


if __name__ == "__main__":
    main()
