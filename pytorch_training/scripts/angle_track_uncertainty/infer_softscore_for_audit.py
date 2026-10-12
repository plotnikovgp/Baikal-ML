#!/usr/bin/env python3
"""CPU audit inference for five-feature or six-feature direction/point models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from infer_rethresholded_for_audit import FIELDS, selected_batch
from softscore_model import upgrade_track_model
from train_mixed_universal import model_from_checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--angle-checkpoint", type=Path, required=True)
    parser.add_argument("--track-checkpoint", type=Path, required=True)
    parser.add_argument("--norm-data", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--start-event", type=int, default=0)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--event-stride", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if not 0.7 <= args.threshold <= 1.0:
        raise ValueError("Threshold must be in [0.70, 1.00]")
    if min(args.batch_size, args.max_seq_len, args.event_stride, args.threads) < 1:
        raise ValueError("Batch size, sequence length, stride, and threads must be positive")
    if args.start_event < 0 or (args.max_events is not None and args.max_events < 1):
        raise ValueError("Start event must be nonnegative and max events positive")
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    scale = checkpoint["args"].get("anchor_scale_m", 50.0)
    model = model_from_checkpoints(args.angle_checkpoint, args.track_checkpoint, scale, "cpu")
    score_feature = bool(checkpoint["args"].get("score_feature", False))
    if score_feature:
        model = upgrade_track_model(model, checkpoint["args"].get("score_mode", "absolute"))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    with h5py.File(args.norm_data, "r") as h5:
        prep = {
            "input_norm_mean": np.asarray(h5["norm_param/mean"]).tolist(),
            "input_norm_std": np.asarray(h5["norm_param/std"]).tolist(),
            "max_seq_len": args.max_seq_len,
            "center_time": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept_total = 0
    with h5py.File(args.data, "r") as h5, args.output.open("w", newline="") as stream:
        group = h5[args.split]
        for key in ("signal_probability/data", "channels/data"):
            if key not in group:
                raise KeyError(f"Missing {args.split}/{key} in {args.data}")
        stored_mean = np.asarray(h5["norm_param/mean"], dtype=np.float32)
        stored_std = np.asarray(h5["norm_param/std"], dtype=np.float32)
        total = len(group["ev_starts/data"]) - 1
        if args.start_event >= total:
            raise ValueError(f"Start event {args.start_event} is outside {total} events")
        end_event = (
            total if args.max_events is None else min(total, args.start_event + args.max_events)
        )
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for begin in range(args.start_event, end_event, args.batch_size):
            end = min(begin + args.batch_size, end_event)
            kept, x, mask = selected_batch(
                group,
                begin,
                end,
                args.threshold,
                prep,
                stored_mean,
                stored_std,
                36,
                args.start_event,
                args.event_stride,
                score_feature=score_feature,
            )
            if not kept:
                continue
            with torch.inference_mode():
                direction, offset = model(torch.from_numpy(x), torch.from_numpy(mask))
            direction = direction.numpy()
            offset = offset.numpy()
            theta = np.rad2deg(np.arccos(np.clip(direction[:, 2], -1, 1)))
            phi = np.mod(np.rad2deg(np.arctan2(direction[:, 1], direction[:, 0])), 360)
            for i, row in enumerate(kept):
                anchor = row[4] + offset[i]
                writer.writerow(
                    {
                        "event_id": row[0],
                        "source_event_index": row[1],
                        "n_hits": row[2],
                        "n_strings": row[3],
                        "theta_deg": float(theta[i]),
                        "phi_deg": float(phi[i]),
                        "direction_x": float(direction[i, 0]),
                        "direction_y": float(direction[i, 1]),
                        "direction_z": float(direction[i, 2]),
                        "anchor_x_m": float(anchor[0]),
                        "anchor_y_m": float(anchor[1]),
                        "anchor_z_m": float(anchor[2]),
                    }
                )
            kept_total += len(kept)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_events": end_event - args.start_event,
                "kept_events": kept_total,
                "threshold": args.threshold,
            }
        )
    )


if __name__ == "__main__":
    main()
