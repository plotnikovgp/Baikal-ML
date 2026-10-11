#!/usr/bin/env python3
"""Infer an unbundled direction/point checkpoint without stale uncertainty heads."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from bundle_universal_model import build_model
from infer_unified_model import read_batch

FIELDS = (
    "event_id",
    "n_hits",
    "n_strings",
    "theta_deg",
    "phi_deg",
    "direction_x",
    "direction_y",
    "direction_z",
    "anchor_x_m",
    "anchor_y_m",
    "anchor_z_m",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--track-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError("batch-size and threads must be positive")
    torch.set_num_threads(args.threads)
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
    checkpoint = torch.load(args.track_checkpoint, map_location="cpu", weights_only=False)
    model = build_model(bundle["model_config"]).base
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    prep = bundle["preprocessing"]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.data, "r") as h5, args.output.open("w", newline="") as stream:
        if args.split not in h5:
            raise KeyError(f"Missing split {args.split!r} in {args.data}")
        total = len(h5[f"{args.split}/ev_starts/data"]) - 1
        if args.max_events is not None:
            total = min(total, args.max_events)
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for begin in range(0, total, args.batch_size):
            end = min(begin + args.batch_size, total)
            ids, lengths, strings, x, mask, centers = read_batch(h5, args.split, begin, end, prep)
            with torch.inference_mode():
                direction, offset = model(torch.from_numpy(x), torch.from_numpy(mask))
            direction = direction.numpy()
            anchor = centers + offset.numpy()
            theta = np.rad2deg(np.arccos(np.clip(direction[:, 2], -1, 1)))
            phi = np.mod(np.rad2deg(np.arctan2(direction[:, 1], direction[:, 0])), 360)
            for i, event_id in enumerate(ids):
                writer.writerow(
                    {
                        "event_id": event_id.decode()
                        if isinstance(event_id, bytes)
                        else str(event_id),
                        "n_hits": int(lengths[i]),
                        "n_strings": int(strings[i]),
                        "theta_deg": float(theta[i]),
                        "phi_deg": float(phi[i]),
                        "direction_x": float(direction[i, 0]),
                        "direction_y": float(direction[i, 1]),
                        "direction_z": float(direction[i, 2]),
                        "anchor_x_m": float(anchor[i, 0]),
                        "anchor_y_m": float(anchor[i, 1]),
                        "anchor_z_m": float(anchor[i, 2]),
                    }
                )
    print(json.dumps({"output": str(args.output), "events": total}))


if __name__ == "__main__":
    main()
