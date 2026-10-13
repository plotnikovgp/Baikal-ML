#!/usr/bin/env python3
"""Per-event error and observed-hit audit for the leakage-free v4 MC test.

The model sees exactly the same selected hits as in normal inference. MC-only
labels and residuals are read strictly for after-the-fact diagnosis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from audit_photon_time_model import photon_time_residual
from train_mixed_universal import AdaptedTrackDataset, model_from_checkpoints


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.rad2deg(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def line_distance_m(point: np.ndarray, truth: np.ndarray, direction: np.ndarray) -> float:
    delta = point - truth
    return float(np.linalg.norm(delta - np.dot(delta, direction) * direction))


def observed_span_m(positions: np.ndarray) -> float:
    difference = positions[:, None] - positions[None]
    return float(np.sqrt(np.sum(difference**2, axis=-1)).max())


def safe_quantile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile)) if len(values) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-data", type=Path, required=True)
    parser.add_argument("--pred-anchors", type=Path, required=True)
    parser.add_argument("--norm-data", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--angle-checkpoint", type=Path, required=True)
    parser.add_argument("--track-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-events", type=int, default=None)
    args = parser.parse_args()

    with h5py.File(args.norm_data, "r") as h5:
        mean = np.asarray(h5["norm_param/mean"], dtype=np.float32)
        std = np.asarray(h5["norm_param/std"], dtype=np.float32)
    with np.load(args.masks) as masks:
        keep = np.asarray(masks[f"pred_{args.split}"], dtype=bool).copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_from_checkpoints(args.angle_checkpoint, args.track_checkpoint, 50.0, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    data = AdaptedTrackDataset(
        str(args.pred_data),
        args.split,
        args.batch_size,
        128,
        False,
        None,
        False,
        anchor_path=str(args.pred_anchors),
        keep=keep,
        expected_mean=mean,
        expected_std=std,
        single_muatm_only=True,
    )
    rows = []
    with (
        h5py.File(args.pred_data, "r", swmr=True) as h5,
        h5py.File(args.pred_anchors, "r", swmr=True) as anchors,
    ):
        group = h5[args.split]
        anchor = anchors[args.split]
        all_ids = np.asarray(group["ev_ids/data"])
        all_valid = np.asarray(anchor["valid"], dtype=bool)
        prefixes = all_ids.astype("S5")
        selection = keep & ((prefixes != b"muatm") | all_valid)
        with torch.inference_mode():
            for batch_index in range(len(data)):
                event_start = batch_index * args.batch_size
                event_end = min(event_start + args.batch_size, data.n_events)
                selected = np.arange(event_start, event_end)[selection[event_start:event_end]]
                x, true_direction, mask, particle, true_point, valid = data[batch_index]
                if len(x) != len(selected):
                    raise AssertionError(
                        f"Selection mismatch batch {batch_index}: {len(x)} vs {len(selected)}"
                    )
                if not len(x):
                    continue
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    pred_direction, pred_point = model(x.to(device), mask.to(device))
                pred_direction = pred_direction.float().cpu().numpy()
                pred_point = pred_point.float().cpu().numpy()
                true_direction = true_direction.numpy()
                true_point = true_point.numpy()
                physical = x.numpy() * std + mean
                starts = np.asarray(
                    group["ev_starts/data"][event_start : event_end + 1], dtype=np.int64
                )
                a0, b0 = int(starts[0]), int(starts[-1])
                label_flat = np.asarray(group["labels/data"][a0:b0])
                p_flat = np.asarray(group["signal_probability/data"][a0:b0])
                t_res_flat = np.asarray(group["t_res/data"][a0:b0])
                primary = np.asarray(group["prime_prty/data"][event_start:event_end])
                strings = np.asarray(group["num_un_strings/data"][event_start:event_end])
                line_centers = np.asarray(anchor["hit_centroid_m"][selected], dtype=np.float64)
                for j, event_index in enumerate(selected):
                    if args.max_events is not None and len(rows) >= args.max_events:
                        break
                    local = int(event_index - event_start)
                    length = int(mask[j].sum())
                    flat_a = int(starts[local] - a0)
                    flat_b = flat_a + length
                    hit = physical[j, :length]
                    xyz = hit[:, 2:5].astype(np.float64)
                    t_ns = hit[:, 1].astype(np.float64)
                    label = label_flat[flat_a:flat_b]
                    p = p_flat[flat_a:flat_b]
                    t_res = t_res_flat[flat_a:flat_b]
                    if not (len(hit) == len(label) == len(p) == len(t_res)):
                        raise AssertionError("Hit metadata length mismatch")
                    center = xyz.mean(axis=0)
                    line_center = line_centers[j]
                    direction = pred_direction[j].astype(np.float64)
                    truth = true_direction[j].astype(np.float64)
                    pred_line = line_center + pred_point[j]
                    truth_line = line_center + true_point[j]
                    pred_photon_residual = photon_time_residual(
                        xyz, t_ns, pred_line, direction, 1.36
                    )
                    truth_photon_residual = (
                        photon_time_residual(xyz, t_ns, truth_line, truth, 1.36)
                        if valid[j]
                        else np.full(length, np.nan)
                    )
                    prefix = bytes(all_ids[event_index])[:5]
                    particle_name = {
                        b"muatm": "muatm",
                        b"nuatm": "nuatm",
                        b"nue2_": "nue2",
                    }.get(prefix, "unknown")
                    rows.append(
                        {
                            "event_id": bytes(all_ids[event_index]).decode(),
                            "event_index": int(event_index),
                            "particle": particle_name,
                            "angular_error_deg": angle_deg(direction, truth),
                            "theta_error_abs_deg": abs(
                                float(np.rad2deg(np.arccos(np.clip(direction[2], -1, 1))))
                                - float(np.rad2deg(np.arccos(np.clip(truth[2], -1, 1))))
                            ),
                            "true_theta_deg": float(primary[local, 0]),
                            "pred_theta_deg": float(
                                np.rad2deg(np.arccos(np.clip(direction[2], -1, 1)))
                            ),
                            "line_error_m": (
                                line_distance_m(pred_point[j], true_point[j], truth)
                                if valid[j]
                                else np.nan
                            ),
                            "anchor_valid": bool(valid[j]),
                            "n_hits": length,
                            "n_strings": int(strings[local]),
                            "energy_primary": float(primary[local, 2]),
                            "primary_muons_n": float(primary[local, 4]),
                            "charge_median": float(np.median(hit[:, 0])),
                            "charge_q90": safe_quantile(hit[:, 0], 0.9),
                            "time_span_ns": float(np.ptp(t_ns)),
                            "time_std_ns": float(np.std(t_ns)),
                            "z_span_m": float(np.ptp(xyz[:, 2])),
                            "z_std_m": float(np.std(xyz[:, 2])),
                            "xy_rms_m": float(
                                np.sqrt(np.mean(np.sum((xyz[:, :2] - center[:2]) ** 2, axis=1)))
                            ),
                            "hit_span_m": observed_span_m(xyz),
                            "p_mean": float(np.mean(p)),
                            "p_min": float(np.min(p)),
                            "p_below_08": float(np.mean(p < 0.8)),
                            "label_negative_fraction": float(np.mean(label < 0)),
                            "label_positive_fraction": float(np.mean(label > 0)),
                            "label_zero_fraction": float(np.mean(label == 0)),
                            "tres_abs_median_ns": float(np.median(np.abs(t_res))),
                            "tres_abs_gt10_fraction": float(np.mean(np.abs(t_res) > 10)),
                            "photon_mad_pred_ns": float(np.median(np.abs(pred_photon_residual))),
                            "photon_mad_truth_ns": float(
                                np.nanmedian(np.abs(truth_photon_residual))
                            )
                            if valid[j]
                            else np.nan,
                        }
                    )
                if batch_index % 100 == 0:
                    print(f"batch={batch_index} events={len(rows)}", flush=True)
                if args.max_events is not None and len(rows) >= args.max_events:
                    break
    frame = pd.DataFrame(rows)
    duplicate_rows = int(frame.event_id.duplicated().sum())
    unique = frame.drop_duplicates("event_id", keep="first")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    summary = {
        "split": args.split,
        "count": len(frame),
        "unique_event_count": len(unique),
        "duplicate_rows": duplicate_rows,
        "angle_q50_q68_q95_deg_unique": [
            safe_quantile(unique.angular_error_deg.to_numpy(), q) for q in (0.5, 0.68, 0.95)
        ],
        "by_particle": {
            name: {
                "count": len(part),
                "angle_q50_q68_q95_deg": [
                    safe_quantile(part.angular_error_deg.to_numpy(), q) for q in (0.5, 0.68, 0.95)
                ],
            }
            for name, part in unique.groupby("particle")
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
