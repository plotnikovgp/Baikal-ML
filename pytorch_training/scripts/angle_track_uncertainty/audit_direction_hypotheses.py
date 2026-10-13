#!/usr/bin/env python3
"""Validation-only check of physical timing as a selector of azimuth alternatives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from audit_photon_time_model import photon_time_residual
from bundle_universal_model import build_model
from train_mixed_universal import AdaptedTrackDataset


def alternatives(direction: np.ndarray) -> np.ndarray:
    x, y, z = direction
    return np.asarray(
        ((x, y, z), (-x, -y, z), (-y, x, z), (y, -x, z)),
        dtype=np.float64,
    )


def summarize(frame: pd.DataFrame, field: str) -> dict[str, float]:
    angle = frame[field].to_numpy()
    return {
        "events": len(angle),
        "q50_deg": float(np.quantile(angle, 0.5)),
        "q68_deg": float(np.quantile(angle, 0.68)),
        "q95_deg": float(np.quantile(angle, 0.95)),
        "severe20_fraction": float(np.mean(angle >= 20)),
        "severe45_fraction": float(np.mean(angle >= 45)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--pred-data", type=Path, required=True)
    parser.add_argument("--pred-anchors", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-events", type=int)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
    full_model = build_model(bundle["model_config"])
    full_model.load_state_dict(bundle["state_dict"], strict=True)
    model = full_model.base.to(device).eval()
    mean = np.asarray(bundle["preprocessing"]["input_norm_mean"], dtype=np.float32)
    std = np.asarray(bundle["preprocessing"]["input_norm_std"], dtype=np.float32)
    with np.load(args.masks) as masks:
        keep = np.asarray(masks["pred_val"], dtype=bool).copy()
    data = AdaptedTrackDataset(
        str(args.pred_data),
        "val",
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
    with torch.inference_mode():
        for batch_index in range(len(data)):
            x, truth, mask, particle, point, valid = data[batch_index]
            if not len(x):
                continue
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                direction, pred_point = model(x.to(device), mask.to(device))
            direction = direction.float().cpu().numpy()
            pred_point = pred_point.float().cpu().numpy()
            physical = x.numpy() * std + mean
            for i in range(len(x)):
                hit = physical[i, mask[i].numpy()]
                xyz = hit[:, 2:5].astype(np.float64)
                time_ns = hit[:, 1].astype(np.float64)
                line_point = xyz.mean(axis=0) + pred_point[i]
                choices = alternatives(direction[i])
                true_direction = truth[i].numpy().astype(np.float64)
                angle = np.rad2deg(np.arccos(np.clip(choices @ true_direction, -1.0, 1.0)))
                mad = np.asarray(
                    [
                        np.median(
                            np.abs(photon_time_residual(xyz, time_ns, line_point, candidate, 1.36))
                        )
                        for candidate in choices
                    ]
                )
                row = {
                    "particle": ("muatm", "nuatm", "nue2")[int(particle[i])],
                    "n_hits": len(hit),
                    "span_m": float(np.linalg.norm(xyz[:, None] - xyz[None], axis=2).max()),
                    "baseline_deg": angle[0],
                    "baseline_mad_ns": mad[0],
                    "oracle2_deg": float(np.min(angle[:2])),
                    "oracle4_deg": float(np.min(angle)),
                }
                for k in (2, 4):
                    for penalty in (0, 2, 5, 10, 20):
                        scores = mad[:k].copy()
                        scores[1:] += penalty
                        choice = int(np.argmin(scores))
                        row[f"select{k}_penalty{penalty}_deg"] = angle[choice]
                        row[f"select{k}_penalty{penalty}_changed"] = choice != 0
                rows.append(row)
                if args.max_events is not None and len(rows) >= args.max_events:
                    break
            if batch_index % 10 == 0:
                print(f"batch={batch_index} events={len(rows)}", flush=True)
            if args.max_events is not None and len(rows) >= args.max_events:
                break
    frame = pd.DataFrame(rows)
    columns = [
        "baseline_deg",
        "oracle2_deg",
        "oracle4_deg",
        *(f"select{k}_penalty{penalty}_deg" for k in (2, 4) for penalty in (0, 2, 5, 10, 20)),
    ]
    groups = {
        "all": frame,
        "span_lt120": frame.loc[frame.span_m < 120],
        "span_120_200": frame.loc[(frame.span_m >= 120) & (frame.span_m < 200)],
        "span_ge200": frame.loc[frame.span_m >= 200],
        **{name: frame.loc[frame.particle == name] for name in ("muatm", "nuatm", "nue2")},
    }
    report = {
        name: {field: summarize(group, field) for field in columns}
        for name, group in groups.items()
    }
    for name, group in groups.items():
        report[name]["selection_changed_fraction"] = {
            f"k{k}_penalty{penalty}": float(group[f"select{k}_penalty{penalty}_changed"].mean())
            for k in (2, 4)
            for penalty in (0, 2, 5, 10, 20)
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "events.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"events": len(frame), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
