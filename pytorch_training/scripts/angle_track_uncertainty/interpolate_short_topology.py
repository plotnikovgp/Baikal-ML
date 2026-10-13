#!/usr/bin/env python3
"""Validation-only weight interpolation of v4 and the short-topology fine-tune."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from bundle_universal_model import build_model
from train_photon_residual import data_loader
from train_short_topology import validation, validation_spans


def blend_state(base: dict, tuned: dict, alpha: float) -> dict:
    if base.keys() != tuned.keys():
        raise ValueError("Model state keys differ")
    result = {}
    for key, original in base.items():
        adapted = tuned[key]
        if original.shape != adapted.shape:
            raise ValueError(f"Model state shape differs for {key}")
        result[key] = (
            original + alpha * (adapted - original) if original.is_floating_point() else original
        )
    return result


def score(result: dict) -> float:
    return (
        result["short_pred_val"]["q68_deg"]
        + 3.0 * result["pred_val"]["direction_deg"]["q68"]
        + 0.5 * result["gt_val_clean"]["direction_deg"]["q68"]
        + 0.1 * result["pred_val"]["muatm"]["direction_deg"]["q68"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--fine-tuned", type=Path, required=True)
    parser.add_argument("--pred-data", type=Path, required=True)
    parser.add_argument("--pred-anchors", type=Path, required=True)
    parser.add_argument("--gt-data", type=Path, required=True)
    parser.add_argument("--gt-anchors", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gt-val-stride", type=int, default=6)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
    full_model = build_model(bundle["model_config"])
    full_model.load_state_dict(bundle["state_dict"], strict=True)
    model = full_model.base
    original = {key: value.cpu().clone() for key, value in model.state_dict().items()}
    checkpoint = torch.load(args.fine_tuned, map_location="cpu", weights_only=False)
    adapted = checkpoint["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    mean = np.asarray(bundle["preprocessing"]["input_norm_mean"], dtype=np.float32)
    std = np.asarray(bundle["preprocessing"]["input_norm_std"], dtype=np.float32)
    with np.load(args.masks) as masks:
        pred_keep = np.asarray(masks["pred_val"], dtype=bool).copy()
        gt_keep = np.asarray(masks["gt_val"], dtype=bool).copy()
    gt_keep &= np.arange(len(gt_keep)) % args.gt_val_stride == 0
    pred_loader = data_loader(
        args.pred_data,
        args.pred_anchors,
        "val",
        args.batch_size,
        128,
        mean,
        std,
        args.workers,
        811,
        keep=pred_keep,
        single_muatm_only=True,
    )
    gt_loader = data_loader(
        args.gt_data,
        args.gt_anchors,
        "val",
        args.batch_size,
        128,
        mean,
        std,
        args.workers,
        812,
        keep=gt_keep,
    )
    spans = validation_spans(pred_loader, torch.from_numpy(mean), torch.from_numpy(std))
    rows = []
    best = None
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        state = blend_state(original, adapted, alpha)
        model.load_state_dict(state, strict=True)
        result = validation(model, pred_loader, gt_loader, spans, device)
        result["alpha"] = alpha
        result["score"] = score(result)
        if rows:
            baseline = rows[0]
            result["admissible"] = (
                result["pred_val"]["direction_deg"]["q68"]
                <= baseline["pred_val"]["direction_deg"]["q68"]
                and result["gt_val_clean"]["direction_deg"]["q68"]
                <= baseline["gt_val_clean"]["direction_deg"]["q68"] + 0.03
                and result["pred_val"]["muatm"]["direction_deg"]["q68"]
                <= baseline["pred_val"]["muatm"]["direction_deg"]["q68"] + 0.10
                and result["short_pred_val"]["q68_deg"]
                <= baseline["short_pred_val"]["q68_deg"] - 0.05
            )
        else:
            result["admissible"] = False
        rows.append(result)
        print("validation " + json.dumps(result, sort_keys=True), flush=True)
        if result["admissible"] and (best is None or result["score"] < best["score"]):
            best = result
            torch.save(
                {"model": state, "alpha": alpha, "metrics": result},
                args.output_dir / "best.pt",
            )
    (args.output_dir / "validation.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"best_alpha": None if best is None else best["alpha"]}), flush=True)


if __name__ == "__main__":
    main()
