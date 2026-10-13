#!/usr/bin/env python3
"""Paired, unique-event-ID test of the photon-residual head against v4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bundle_universal_model import build_model
from photon_residual_upgrade import PhotonResidualUpgrade
from train_angle_signal import make_loader
from train_mixed_universal import AdaptedTrackDataset, evaluate


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {f"q{int(q * 100)}": float(np.quantile(values, q)) for q in (0.5, 0.68, 0.95)}


def metrics(frame: pd.DataFrame, field: str) -> dict:
    angle = frame[field].to_numpy()
    point = frame[f"{field}_line"].to_numpy()
    return {
        "count": len(frame),
        "angle_deg": quantiles(angle),
        "line_m": quantiles(point[np.isfinite(point)]),
        "error_ge20": int((angle >= 20).sum()),
        "error_ge45": int((angle >= 45).sum()),
        "error_ge90": int((angle >= 90).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pred-data", type=Path, required=True)
    parser.add_argument("--pred-anchors", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--baseline-events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
    full_model = build_model(bundle["model_config"])
    full_model.load_state_dict(bundle["state_dict"], strict=True)
    mean = np.asarray(bundle["preprocessing"]["input_norm_mean"], dtype=np.float32)
    std = np.asarray(bundle["preprocessing"]["input_norm_std"], dtype=np.float32)
    # The trusted training checkpoint also stores argparse Path objects.
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PhotonResidualUpgrade(
        full_model.base,
        torch.from_numpy(mean),
        torch.from_numpy(std),
        hidden=saved["args"]["hidden"],
    ).to(device)
    model.load_state_dict(saved["model"], strict=True)
    with np.load(args.masks) as masks:
        keep = np.asarray(masks["pred_test"], dtype=bool).copy()
    data = AdaptedTrackDataset(
        str(args.pred_data),
        "test",
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
    loader = make_loader(data, 0, False, 743)
    _, prediction = evaluate(model, loader, device, return_arrays=True)
    baseline = pd.read_csv(args.baseline_events)
    if len(baseline) != len(prediction["angle"]):
        raise AssertionError(f"Row count mismatch: {len(baseline)} vs {len(prediction['angle'])}")
    particle = np.asarray([{"muatm": 0, "nuatm": 1, "nue2": 2}[name] for name in baseline.particle])
    if not np.array_equal(particle, prediction["particle"]):
        raise AssertionError("Particle order mismatch with baseline audit")
    baseline["baseline_angle"] = baseline.angular_error_deg
    baseline["baseline_angle_line"] = baseline.line_error_m
    baseline["candidate_angle"] = prediction["angle"]
    baseline["candidate_angle_line"] = np.where(prediction["valid"], prediction["line"], np.nan)
    unique = baseline.drop_duplicates("event_id", keep="first").copy()
    report = {
        "checkpoint": str(args.checkpoint),
        "rows": len(baseline),
        "unique_events": len(unique),
        "duplicate_rows": len(baseline) - len(unique),
        "selection": "predicted-signal p>=0.70, 2 strings/8 hits, single-muatm, independent pred_test mask",
        "all": {},
        "by_particle": {},
        "by_span": {},
    }
    groups = {
        "all": unique,
        **{name: unique.loc[unique.particle == name] for name in ("muatm", "nuatm", "nue2")},
        "span_lt100": unique.loc[unique.hit_span_m < 100],
        "span_100_150": unique.loc[(unique.hit_span_m >= 100) & (unique.hit_span_m < 150)],
        "span_150_200": unique.loc[(unique.hit_span_m >= 150) & (unique.hit_span_m < 200)],
        "span_ge200": unique.loc[unique.hit_span_m >= 200],
    }
    for name, frame in groups.items():
        result = {
            "baseline": metrics(frame, "baseline_angle"),
            "candidate": metrics(frame, "candidate_angle"),
            "rescued_ge20": int(
                ((frame.baseline_angle >= 20) & (frame.candidate_angle < 20)).sum()
            ),
            "new_ge20": int(((frame.baseline_angle < 20) & (frame.candidate_angle >= 20)).sum()),
        }
        if name == "all":
            report["all"] = result
        elif name.startswith("span_"):
            report["by_span"][name] = result
        else:
            report["by_particle"][name] = result
    unique.to_csv(args.output_dir / "paired_unique_events.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
