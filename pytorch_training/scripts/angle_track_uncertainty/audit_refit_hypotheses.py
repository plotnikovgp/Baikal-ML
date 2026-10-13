#!/usr/bin/env python3
"""Check whether fitting a point on each fixed-azimuth hypothesis aids timing selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from audit_direction_hypotheses import alternatives
from bundle_universal_model import build_model
from fit_photon_track import photon_model, transverse_basis
from scipy.optimize import least_squares
from train_mixed_universal import AdaptedTrackDataset


def refit_point(
    xyz: np.ndarray,
    times: np.ndarray,
    start: np.ndarray,
    direction: np.ndarray,
    *,
    point_prior_m: float = 20.0,
    time_scale_ns: float = 10.0,
) -> tuple[float, float, float]:
    e1, e2 = transverse_basis(direction)
    basis = np.stack((e1, e2), axis=0)

    def centered_residual(params: np.ndarray) -> np.ndarray:
        point = start + point_prior_m * (params @ basis)
        residual = times - photon_model(xyz, point, direction, 1.36)
        return residual - np.median(residual)

    def objective(params: np.ndarray) -> np.ndarray:
        residual = centered_residual(params)
        return np.concatenate((residual / time_scale_ns, params))

    result = least_squares(
        objective,
        np.zeros(2),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=35,
        bounds=(-3.0, 3.0),
    )
    residual = centered_residual(result.x)
    mad = float(np.median(np.abs(residual)))
    robust = float(np.mean(2 * (np.sqrt(1 + (residual / time_scale_ns) ** 2) - 1)))
    shift = float(point_prior_m * np.linalg.norm(result.x))
    return mad, robust, shift


def metrics(frame: pd.DataFrame, field: str) -> dict:
    x = frame[field].to_numpy()
    return {
        "events": len(x),
        "q50_deg": float(np.quantile(x, 0.5)),
        "q68_deg": float(np.quantile(x, 0.68)),
        "q95_deg": float(np.quantile(x, 0.95)),
        "severe20_fraction": float(np.mean(x >= 20)),
        "severe45_fraction": float(np.mean(x >= 45)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--pred-data", type=Path, required=True)
    parser.add_argument("--pred-anchors", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--span-max-m", type=float, default=120.0)
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
        128,
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
            x, truth, mask, particle, _point, _valid = data[batch_index]
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
                centered_xyz = xyz - xyz.mean(axis=0)
                span = float(np.linalg.norm(xyz[:, None] - xyz[None], axis=2).max())
                if span >= args.span_max_m:
                    continue
                times = hit[:, 1].astype(np.float64)
                choices = alternatives(direction[i])
                truth_i = truth[i].numpy().astype(np.float64)
                angle = np.rad2deg(np.arccos(np.clip(choices @ truth_i, -1.0, 1.0)))
                fits = np.asarray(
                    [refit_point(centered_xyz, times, pred_point[i], u) for u in choices]
                )
                row = {
                    "particle": ("muatm", "nuatm", "nue2")[int(particle[i])],
                    "span_m": span,
                    "n_hits": len(hit),
                    "baseline_deg": angle[0],
                    "oracle4_deg": float(np.min(angle)),
                    "baseline_fit_mad_ns": fits[0, 0],
                }
                for k in (2, 4):
                    for score_field, score_index in (("mad", 0), ("robust", 1)):
                        for alternative_penalty in (0, 2, 5, 10, 20):
                            score = fits[:k, score_index].copy()
                            score[1:] += alternative_penalty * (1 if score_field == "mad" else 0.1)
                            choice = int(np.argmin(score))
                            row[f"select{k}_{score_field}_penalty{alternative_penalty}_deg"] = (
                                angle[choice]
                            )
                            row[f"select{k}_{score_field}_penalty{alternative_penalty}_changed"] = (
                                choice != 0
                            )
                rows.append(row)
                if args.max_events is not None and len(rows) >= args.max_events:
                    break
            if batch_index % 10 == 0:
                print(f"batch={batch_index} short_events={len(rows)}", flush=True)
            if args.max_events is not None and len(rows) >= args.max_events:
                break
    frame = pd.DataFrame(rows)
    fields = ["baseline_deg", "oracle4_deg"] + [
        f"select{k}_{score}_penalty{penalty}_deg"
        for k in (2, 4)
        for score in ("mad", "robust")
        for penalty in (0, 2, 5, 10, 20)
    ]
    groups = {
        "all_short": frame,
        **{name: frame.loc[frame.particle == name] for name in ("muatm", "nuatm", "nue2")},
    }
    report = {
        name: {field: metrics(group, field) for field in fields} for name, group in groups.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "events.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"events": len(frame), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
