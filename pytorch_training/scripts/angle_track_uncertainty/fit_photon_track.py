#!/usr/bin/env python3
"""Validation-only robust Cherenkov-time refinement of one v4 track prediction.

The neural network supplies a single starting line; this is not an ensemble.
The fit has an event-time nuisance parameter and a prior on changing the line.
It uses only observed hits at inference, never truth. Truth is used for audit.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from audit_photon_time_model import C_M_PER_NS
from scipy.optimize import least_squares
from train_mixed_universal import AdaptedTrackDataset, model_from_checkpoints

PARTICLE = {0: "muatm", 1: "nuatm", 2: "nue2"}


def transverse_basis(unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([0.0, 0.0, 1.0]) if abs(unit[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(unit, reference)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(unit, e1)


def photon_model(
    positions_m: np.ndarray,
    point_m: np.ndarray,
    direction: np.ndarray,
    refractive_index: float,
) -> np.ndarray:
    delta = positions_m - point_m
    along = delta @ direction
    transverse = np.linalg.norm(delta - along[:, None] * direction, axis=1)
    return (along + np.sqrt(refractive_index**2 - 1.0) * transverse) / C_M_PER_NS


def fit_one(
    positions_m: np.ndarray,
    times_ns: np.ndarray,
    initial_direction: np.ndarray,
    initial_offset_m: np.ndarray,
    *,
    refractive_index: float = 1.36,
    time_scale_ns: float = 10.0,
    angle_prior_deg: float = 5.0,
    point_prior_m: float = 10.0,
    max_nfev: int = 50,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if len(times_ns) < 8:
        raise ValueError("At least eight hits are required")
    u0 = np.asarray(initial_direction, dtype=np.float64)
    u0 /= np.linalg.norm(u0)
    p0 = np.asarray(initial_offset_m, dtype=np.float64)
    centered = np.asarray(positions_m, dtype=np.float64) - np.mean(positions_m, axis=0)
    times = np.asarray(times_ns, dtype=np.float64)
    e1, e2 = transverse_basis(u0)
    angle_scale = np.deg2rad(angle_prior_deg)

    def line(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = u0 + angle_scale * (params[0] * e1 + params[1] * e2)
        u /= np.linalg.norm(u)
        p = p0 + point_prior_m * (params[2] * e1 + params[3] * e2)
        return u, p

    starting_model = photon_model(centered, p0, u0, refractive_index)
    initial_t0 = float(np.median(times - starting_model))

    def residual(params: np.ndarray) -> np.ndarray:
        u, p = line(params)
        model = photon_model(centered, p, u, refractive_index)
        t0 = initial_t0 + params[4] * time_scale_ns
        return np.concatenate(((times - model - t0) / time_scale_ns, params[:4]))

    start = np.zeros(5, dtype=np.float64)
    result = least_squares(
        residual,
        start,
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=max_nfev,
        bounds=([-4.0] * 4 + [-100.0], [4.0] * 4 + [100.0]),
    )
    u, p = line(result.x)
    initial_residual = times - starting_model
    initial_residual -= np.median(initial_residual)
    final_residual = times - photon_model(centered, p, u, refractive_index)
    final_residual -= np.median(final_residual)
    diagnostics = {
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "initial_mad_ns": float(np.median(np.abs(initial_residual))),
        "final_mad_ns": float(np.median(np.abs(final_residual))),
        "direction_shift_deg": float(np.rad2deg(np.arccos(np.clip(u @ u0, -1.0, 1.0)))),
        "point_shift_m": float(np.linalg.norm(p - p0)),
    }
    return u, p, diagnostics


def angular_error(direction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.rad2deg(np.arccos(np.clip(direction @ truth, -1.0, 1.0))))


def line_error(point: np.ndarray, truth_point: np.ndarray, truth_direction: np.ndarray) -> float:
    delta = point - truth_point
    return float(np.linalg.norm(delta - (delta @ truth_direction) * truth_direction))


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "count": int(len(values)),
        "q50": float(np.quantile(values, 0.5)),
        "q68": float(np.quantile(values, 0.68)),
        "q90": float(np.quantile(values, 0.9)),
    }


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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--refractive-index", type=float, default=1.36)
    parser.add_argument("--time-scale-ns", type=float, default=10.0)
    parser.add_argument("--angle-prior-deg", type=float, default=5.0)
    parser.add_argument("--point-prior-m", type=float, default=10.0)
    parser.add_argument("--max-nfev", type=int, default=50)
    args = parser.parse_args()

    with h5py.File(args.norm_data, "r") as h5:
        mean = np.asarray(h5["norm_param/mean"], dtype=np.float32)
        std = np.asarray(h5["norm_param/std"], dtype=np.float32)
    with np.load(args.masks) as masks:
        keep = masks["pred_val"].copy()
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_from_checkpoints(args.angle_checkpoint, args.track_checkpoint, 50.0, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    rows: dict[str, list[float]] = defaultdict(list)
    started = time.monotonic()
    with torch.inference_mode():
        for batch_index in range(len(data)):
            x, truth_dir, mask, particle, truth_point, valid = data[batch_index]
            if not len(x):
                continue
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                pred_dir, pred_point = model(x.to(device), mask.to(device))
            pred_dir = pred_dir.float().cpu().numpy()
            pred_point = pred_point.float().cpu().numpy()
            physical = x.numpy() * std + mean
            for j in range(len(x)):
                if args.max_events is not None and len(rows["particle"]) >= args.max_events:
                    break
                if not valid[j]:
                    continue
                hit = physical[j, mask[j].numpy()]
                fit_dir, fit_point, diagnostics = fit_one(
                    hit[:, 2:5],
                    hit[:, 1],
                    pred_dir[j],
                    pred_point[j],
                    refractive_index=args.refractive_index,
                    time_scale_ns=args.time_scale_ns,
                    angle_prior_deg=args.angle_prior_deg,
                    point_prior_m=args.point_prior_m,
                    max_nfev=args.max_nfev,
                )
                true_dir = truth_dir[j].numpy()
                true_point = truth_point[j].numpy()
                rows["particle"].append(int(particle[j]))
                rows["hits"].append(len(hit))
                rows["nn_angle_deg"].append(angular_error(pred_dir[j], true_dir))
                rows["fit_angle_deg"].append(angular_error(fit_dir, true_dir))
                rows["nn_line_m"].append(line_error(pred_point[j], true_point, true_dir))
                rows["fit_line_m"].append(line_error(fit_point, true_point, true_dir))
                for name, value in diagnostics.items():
                    rows[name].append(value)
            if args.max_events is not None and len(rows["particle"]) >= args.max_events:
                break
            if batch_index % 10 == 0:
                print(
                    f"batch={batch_index} events={len(rows['particle'])} elapsed_s={time.monotonic() - started:.1f}",
                    flush=True,
                )
    arrays = {name: np.asarray(values) for name, values in rows.items()}
    report = {
        "args": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "elapsed_s": time.monotonic() - started,
        "cohort": "pred_val leak-free, single-track muatm only, GT not used by fit",
        "subsets": {},
    }
    for name, selection in [("all", np.ones(len(arrays["particle"]), dtype=bool))] + [
        (particle_name, arrays["particle"] == code) for code, particle_name in PARTICLE.items()
    ]:
        if not selection.any():
            continue
        report["subsets"][name] = {
            metric: summarize(arrays[metric][selection])
            for metric in (
                "nn_angle_deg",
                "fit_angle_deg",
                "nn_line_m",
                "fit_line_m",
                "initial_mad_ns",
                "final_mad_ns",
                "direction_shift_deg",
                "point_shift_m",
                "nfev",
            )
        }
        report["subsets"][name]["fit_improves_angle_fraction"] = float(
            np.mean(arrays["fit_angle_deg"][selection] < arrays["nn_angle_deg"][selection])
        )
        report["subsets"][name]["fit_improves_line_fraction"] = float(
            np.mean(arrays["fit_line_m"][selection] < arrays["nn_line_m"][selection])
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
