#!/usr/bin/env python3
"""Validation-only sweep of one photon-residual model's direction correction scale.

This changes a scalar in the same trained single model, not a model ensemble.
No test or experimental truth is used to choose the scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from bundle_universal_model import build_model
from photon_residual_upgrade import PhotonResidualUpgrade
from train_mixed_universal import evaluate, validation_score
from train_photon_residual import data_loader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pred-data", type=Path, required=True)
    parser.add_argument("--pred-anchors", type=Path, required=True)
    parser.add_argument("--gt-data", type=Path, required=True)
    parser.add_argument("--gt-anchors", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scales", type=float, nargs="+", default=(0.0, 0.15, 0.30, 0.45, 0.60))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gt-val-stride", type=int, default=6)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not args.scales or min(args.scales) < 0 or 0.30 not in args.scales:
        raise ValueError("Scales must include nonnegative values and baseline 0.30")
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
    full_model = build_model(bundle["model_config"])
    full_model.load_state_dict(bundle["state_dict"], strict=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mean = np.asarray(bundle["preprocessing"]["input_norm_mean"], dtype=np.float32)
    std = np.asarray(bundle["preprocessing"]["input_norm_std"], dtype=np.float32)
    model = PhotonResidualUpgrade(
        full_model.base,
        torch.from_numpy(mean),
        torch.from_numpy(std),
        hidden=checkpoint["args"]["hidden"],
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    with np.load(args.masks) as masks:
        pred_keep = np.asarray(masks["pred_val"], dtype=bool).copy()
        gt_keep = np.asarray(masks["gt_val"], dtype=bool).copy()
    gt_keep &= np.arange(len(gt_keep)) % args.gt_val_stride == 0
    pred_val = data_loader(
        args.pred_data,
        args.pred_anchors,
        "val",
        args.batch_size,
        128,
        mean,
        std,
        args.workers,
        743,
        keep=pred_keep,
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
        744,
        keep=gt_keep,
    )
    rows = []
    for scale in args.scales:
        model.direction_scale = scale
        result = {
            "direction_scale": scale,
            "pred_val": evaluate(model, pred_val, device),
            "gt_val_clean": evaluate(model, gt_val, device),
        }
        result["score"] = validation_score(result, 0.5, 0.03)
        rows.append(result)
        print(
            json.dumps(
                {
                    "scale": scale,
                    "score": result["score"],
                    "pred_q68": result["pred_val"]["direction_deg"]["q68"],
                    "gt_q68": result["gt_val_clean"]["direction_deg"]["q68"],
                }
            ),
            flush=True,
        )
    baseline = next(row for row in rows if row["direction_scale"] == 0.30)
    pred_limit = baseline["pred_val"]["direction_deg"]["q68"]
    gt_limit = baseline["gt_val_clean"]["direction_deg"]["q68"] + 0.02
    for row in rows:
        row["admissible"] = (
            row["pred_val"]["direction_deg"]["q68"] <= pred_limit
            and row["gt_val_clean"]["direction_deg"]["q68"] <= gt_limit
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"checkpoint": str(args.checkpoint), "rows": rows}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
