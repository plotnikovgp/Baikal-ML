#!/usr/bin/env python3
"""Diagnostic-only GT-label removal of false-positive noise hits on v4 test.

Uses MC labels at inference solely to estimate an upper bound on improving the
signal/noise stage; this is not a deployable pipeline or a fair model metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from train_mixed_universal import model_from_checkpoints


def angular_error(prediction: np.ndarray, theta_phi_deg: np.ndarray) -> float:
    theta, phi = np.deg2rad(theta_phi_deg[:2])
    truth = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    return float(np.rad2deg(np.arccos(np.clip(prediction @ truth, -1.0, 1.0))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--norm-data", type=Path, required=True)
    parser.add_argument("--angle-checkpoint", type=Path, required=True)
    parser.add_argument("--track-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    frame = pd.read_csv(args.events).drop_duplicates("event_id", keep="first")
    possible = frame.loc[frame.label_zero_fraction > 0].copy()
    with h5py.File(args.norm_data, "r") as h5:
        target_mean = np.asarray(h5["norm_param/mean"], dtype=np.float32)
        target_std = np.asarray(h5["norm_param/std"], dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_from_checkpoints(args.angle_checkpoint, args.track_checkpoint, 50.0, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    eligible = []
    with h5py.File(args.data, "r") as h5:
        group = h5["test"]
        source_mean = np.asarray(h5["norm_param/mean"], dtype=np.float32)
        source_std = np.asarray(h5["norm_param/std"], dtype=np.float32)
        starts = group["ev_starts/data"]
        for row in possible.itertuples(index=False):
            index = int(row.event_index)
            event_id = bytes(group["ev_ids/data"][index]).decode()
            if event_id != row.event_id:
                raise ValueError(f"Event ID mismatch at row {index}")
            a, b = int(starts[index]), int(starts[index + 1])
            labels = np.asarray(group["labels/data"][a:b])[:128]
            channels = np.asarray(group["channels/data"][a:b])[:128]
            keep = labels != 0
            if keep.sum() < 8 or len(np.unique(channels[keep] // 36)) < 2:
                continue
            raw = np.asarray(group["data/data"][a:b], dtype=np.float32)[:128]
            normalized = (raw[keep] * source_std + source_mean - target_mean) / target_std
            true_angles = np.asarray(group["prime_prty/data"][index], dtype=np.float32)
            eligible.append(
                (
                    row.event_id,
                    row.particle,
                    row.angular_error_deg,
                    normalized,
                    true_angles,
                )
            )
    results = []
    with torch.inference_mode():
        for offset in range(0, len(eligible), args.batch_size):
            batch = eligible[offset : offset + args.batch_size]
            x = np.zeros((len(batch), max(len(event[3]) for event in batch), 5), dtype=np.float32)
            mask = np.zeros(x.shape[:2], dtype=bool)
            for j, event in enumerate(batch):
                x[j, : len(event[3])] = event[3]
                mask[j, : len(event[3])] = True
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                direction, _point = model(
                    torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device)
                )
            direction = direction.float().cpu().numpy()
            for j, event in enumerate(batch):
                results.append(
                    {
                        "event_id": event[0],
                        "particle": event[1],
                        "original_angle_deg": event[2],
                        "oracle_no_noise_angle_deg": angular_error(direction[j], event[4]),
                        "retained_hits": len(event[3]),
                    }
                )
    output = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    merged = frame.merge(
        output[["event_id", "oracle_no_noise_angle_deg"]], on="event_id", how="left"
    )
    merged["oracle_or_original_angle_deg"] = merged.oracle_no_noise_angle_deg.fillna(
        merged.angular_error_deg
    )
    summary = {
        "unique_test_events": len(frame),
        "any_selected_noise_events": len(possible),
        "oracle_cut_eligible_events": len(output),
        "original_q50_q68": [float(np.quantile(frame.angular_error_deg, q)) for q in (0.5, 0.68)],
        "oracle_or_original_q50_q68": [
            float(np.quantile(merged.oracle_or_original_angle_deg, q)) for q in (0.5, 0.68)
        ],
        "original_ge20": int((merged.angular_error_deg >= 20).sum()),
        "oracle_or_original_ge20": int((merged.oracle_or_original_angle_deg >= 20).sum()),
        "eligible_original_ge20": int((output.original_angle_deg >= 20).sum()),
        "eligible_oracle_ge20": int((output.oracle_no_noise_angle_deg >= 20).sum()),
        "eligible_oracle_improves_fraction": float(
            (output.oracle_no_noise_angle_deg < output.original_angle_deg).mean()
        )
        if len(output)
        else float("nan"),
        "by_particle": {
            name: {
                "events": len(part),
                "original_ge20": int((part.original_angle_deg >= 20).sum()),
                "oracle_ge20": int((part.oracle_no_noise_angle_deg >= 20).sum()),
                "original_q68": float(np.quantile(part.original_angle_deg, 0.68)),
                "oracle_q68": float(np.quantile(part.oracle_no_noise_angle_deg, 0.68)),
            }
            for name, part in output.groupby("particle")
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
