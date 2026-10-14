#!/usr/bin/env python3
"""Audit predicted reference-line uncertainty on independent 2019 nue2 MC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_line_error(predictions_path: Path, truth_points_path: Path):
    predictions = pd.read_csv(predictions_path)
    with np.load(truth_points_path) as truth:
        ids = truth["event_ids"].astype(str)
        valid = truth["valid"].astype(bool)
        prime = np.asarray(truth["prime"], dtype=np.float64)
        point = np.asarray(truth["point_m"], dtype=np.float64)
    if not predictions.event_id.is_unique:
        raise ValueError("Prediction event IDs must be unique")
    unique_ids, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    del unique_ids
    valid &= counts[inverse] == 1
    available = pd.Index(ids[valid])
    use = predictions.event_id.isin(available).to_numpy()
    predictions = predictions.loc[use].copy()
    positions = available.get_indexer(predictions.event_id.to_numpy(str))
    if (positions < 0).any():
        raise ValueError("Truth point lookup failed")
    truth_rows = np.flatnonzero(valid)[positions]
    theta, phi = np.deg2rad(prime[truth_rows, 0]), np.deg2rad(prime[truth_rows, 1])
    direction = np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )
    predicted_point = predictions[["anchor_x_m", "anchor_y_m", "anchor_z_m"]].to_numpy()
    delta = predicted_point - point[truth_rows]
    transverse = delta - np.einsum("ij,ij->i", delta, direction)[:, None] * direction
    error = np.linalg.norm(transverse, axis=1)
    sigma = predictions.sigma_anchor_m.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(error)) or not np.all(np.isfinite(sigma) & (sigma > 0)):
        raise ValueError("Invalid line errors or uncertainty scales")
    return predictions, error, sigma, int((~use).sum())


def summarize(error: np.ndarray, sigma: np.ndarray, factors: dict[str, float]):
    return {
        "events": int(len(error)),
        "line_q50_m": float(np.quantile(error, 0.5)),
        "line_q68_m": float(np.quantile(error, 0.68)),
        "line_q95_m": float(np.quantile(error, 0.95)),
        "coverage68": float(np.mean(error <= sigma * factors["68"])),
        "coverage95": float(np.mean(error <= sigma * factors["95"])),
        "required_factor68": float(np.quantile(error / sigma, 0.68)),
        "required_factor95": float(np.quantile(error / sigma, 0.95)),
        "median_sigma_m": float(np.median(sigma)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--truth-points", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
    factors = bundle["calibration"]["line"]
    predictions, error, sigma, excluded = load_line_error(args.predictions, args.truth_points)
    report = {
        "predictions": str(args.predictions),
        "truth_points": str(args.truth_points),
        "selection": "frozen hit classifier p>=0.70, >=8 selected hits/2 strings; unique valid single truth line",
        "excluded_prediction_rows": excluded,
        "all": summarize(error, sigma, factors),
        "by_hits": [],
        "by_predicted_sigma_decile": [],
    }
    hits = predictions.n_hits.to_numpy(dtype=np.int32)
    for low, high in ((8, 12), (12, 16), (16, 24), (24, 40), (40, 1000000)):
        keep = (hits >= low) & (hits < high)
        report["by_hits"].append(
            {
                "hits_low": low,
                "hits_high_exclusive": high,
                **summarize(error[keep], sigma[keep], factors),
            }
        )
    for part in np.array_split(np.argsort(sigma), 10):
        report["by_predicted_sigma_decile"].append(summarize(error[part], sigma[part], factors))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["all"], indent=2))


if __name__ == "__main__":
    main()
