#!/usr/bin/env python3
"""Infer fixed p>=0.70 exp/muatm-MC cohorts with the one-pass raw-hit model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from infer_rethresholded_for_audit import FIELDS
from joint_raw_dataset import IndexedRawBatchDataset
from joint_raw_model import JointRawReconstructor


@torch.inference_mode()
def infer(args: argparse.Namespace) -> dict:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = JointRawReconstructor(
        legacy_float_mask=not checkpoint["args"].get("hard_padding_mask", False)
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.set_num_threads(args.threads)
    model = model.to(device).eval()
    data = IndexedRawBatchDataset(
        args.raw_data,
        args.index,
        args.split,
        args.model_norm_source,
        args.batch_events,
        args.max_seq_len,
    )
    with np.load(args.index) as selection:
        hits = np.asarray(selection[f"{args.split}_hits"])
        strings = np.asarray(selection[f"{args.split}_strings"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        total = 0
        for batch_index in range(len(data)):
            x, mask, _signal, _direction, _point, _valid, _particle = data[batch_index]
            begin = batch_index * args.batch_events
            x, mask = x.to(device), mask.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                direction, offset, centroid, _logit, _probability = model(x, mask)
            direction = direction.float().cpu().numpy()
            anchor = (centroid + offset).float().cpu().numpy()
            theta = np.rad2deg(np.arccos(np.clip(direction[:, 2], -1.0, 1.0)))
            phi = np.mod(np.rad2deg(np.arctan2(direction[:, 1], direction[:, 0])), 360.0)
            for row in range(len(direction)):
                i = begin + row
                writer.writerow(
                    {
                        "event_id": data.ids[i].decode(),
                        "source_event_index": int(data.indices[i]),
                        "n_hits": int(hits[i]),
                        "n_strings": int(strings[i]),
                        "theta_deg": float(theta[row]),
                        "phi_deg": float(phi[row]),
                        "direction_x": float(direction[row, 0]),
                        "direction_y": float(direction[row, 1]),
                        "direction_z": float(direction[row, 2]),
                        "anchor_x_m": float(anchor[row, 0]),
                        "anchor_y_m": float(anchor[row, 1]),
                        "anchor_z_m": float(anchor[row, 2]),
                    }
                )
            total += len(direction)
    return {"output": str(args.output), "events": total, "split": args.split}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--split", choices=("domain_mc", "exp"), required=True)
    parser.add_argument("--model-norm-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-events", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    print(json.dumps(infer(args)), flush=True)


if __name__ == "__main__":
    main()
