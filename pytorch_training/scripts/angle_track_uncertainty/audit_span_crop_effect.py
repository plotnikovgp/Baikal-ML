#!/usr/bin/env python3
"""Calibrate the train-only z-span crop against the fixed muatm MC cohort."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from span_crop_augmentation import crop_vertical_span
from train_mixed_universal import AdaptedTrackDataset


def z_spread(x: torch.Tensor, mask: torch.Tensor, mean: float, std: float) -> np.ndarray:
    z = x[..., 4] * std + mean
    count = mask.sum(dim=1).clamp_min(1)
    center = (z * mask).sum(dim=1) / count
    spread = torch.sqrt(((z - center[:, None]).square() * mask).sum(dim=1) / count)
    return spread.numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--norm-data", type=Path, required=True)
    parser.add_argument("--max-events", type=int, default=25_972)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--probability", type=float, required=True)
    parser.add_argument("--min-keep", type=float, required=True)
    parser.add_argument("--max-keep", type=float, required=True)
    parser.add_argument("--seed", type=int, default=684)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    with h5py.File(args.norm_data, "r") as h5:
        mean = np.asarray(h5["norm_param/mean"], dtype=np.float32)
        std = np.asarray(h5["norm_param/std"], dtype=np.float32)
    data = AdaptedTrackDataset(
        str(args.data),
        "val",
        args.batch_size,
        128,
        False,
        None,
        False,
        anchor_path=str(args.anchors),
        keep=None,
        expected_mean=mean,
        expected_std=std,
        single_muatm_only=False,
    )
    old_spread, new_spread, old_hits, new_hits = [], [], [], []
    for index in range(math.ceil(args.max_events / args.batch_size)):
        x, direction, mask, _particle, point, valid = data[index]
        if (index + 1) * args.batch_size > args.max_events:
            count = args.max_events - index * args.batch_size
            x, direction, mask, point, valid = (
                value[:count] for value in (x, direction, mask, point, valid)
            )
        new_x, new_mask, _ = crop_vertical_span(
            x,
            mask,
            direction,
            point,
            valid,
            torch.from_numpy(mean),
            torch.from_numpy(std),
            probability=args.probability,
            min_keep_fraction=args.min_keep,
            max_keep_fraction=args.max_keep,
        )
        old_spread.append(z_spread(x, mask, float(mean[4]), float(std[4])))
        new_spread.append(z_spread(new_x, new_mask, float(mean[4]), float(std[4])))
        old_hits.append(mask.sum(dim=1).numpy())
        new_hits.append(new_mask.sum(dim=1).numpy())
    old_spread = np.concatenate(old_spread)
    new_spread = np.concatenate(new_spread)
    old_hits = np.concatenate(old_hits)
    new_hits = np.concatenate(new_hits)
    report = {
        "rows_including_duplicate_event_ids": int(len(old_spread)),
        "probability": args.probability,
        "min_keep_fraction": args.min_keep,
        "max_keep_fraction": args.max_keep,
        "applied_fraction": float(np.mean(new_hits < old_hits)),
        "z_std_old_q50_q68_m": np.quantile(old_spread, [0.5, 0.68]).tolist(),
        "z_std_new_q50_q68_m": np.quantile(new_spread, [0.5, 0.68]).tolist(),
        "hits_old_q50_q68": np.quantile(old_hits, [0.5, 0.68]).tolist(),
        "hits_new_q50_q68": np.quantile(new_hits, [0.5, 0.68]).tolist(),
        "min_new_hits": int(new_hits.min()),
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
