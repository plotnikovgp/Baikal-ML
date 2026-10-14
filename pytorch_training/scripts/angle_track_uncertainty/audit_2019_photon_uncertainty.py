#!/usr/bin/env python3
"""Check directional quality and interval coverage on a separate 2019 MC set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def summarize(error: np.ndarray, radius68: np.ndarray, radius95: np.ndarray) -> dict:
    return {
        "events": int(len(error)),
        "angle_q50_deg": float(np.quantile(error, 0.5)),
        "angle_q68_deg": float(np.quantile(error, 0.68)),
        "angle_q95_deg": float(np.quantile(error, 0.95)),
        "angle_coverage68": float(np.mean(error <= radius68)),
        "angle_coverage95": float(np.mean(error <= radius95)),
        "required_r68_scale_for_68pct": float(np.quantile(error / radius68, 0.68)),
        "required_r95_scale_for_95pct": float(np.quantile(error / radius95, 0.95)),
        "median_r68_deg": float(np.median(radius68)),
        "median_r95_deg": float(np.median(radius95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--mc-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    with h5py.File(args.mc_data) as h5:
        group = h5["test"]
        ids = group["ev_ids/data"][:].astype("U")
        prime = group["prime_prty/data"][:, :2].astype(np.float64)
    if len(predictions) != len(ids) or not np.array_equal(predictions.event_id.to_numpy(), ids):
        raise ValueError("Prediction rows must exactly match HDF5 event IDs and order")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("2019 evaluation IDs are not unique")

    theta, phi = np.deg2rad(prime[:, 0]), np.deg2rad(prime[:, 1])
    truth = np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )
    predicted = predictions[["direction_x", "direction_y", "direction_z"]].to_numpy()
    predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
    error = np.rad2deg(
        np.arctan2(
            np.linalg.norm(np.cross(truth, predicted), axis=1),
            np.einsum("ij,ij->i", truth, predicted),
        )
    )
    r68 = predictions.angle_r68_deg.to_numpy(dtype=np.float64)
    r95 = predictions.angle_r95_deg.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(error)) or not np.all((r68 > 0) & (r95 >= r68)):
        raise ValueError("Nonfinite angular errors or invalid uncertainty radii")

    report = {
        "predictions": str(args.predictions),
        "mc_data": str(args.mc_data),
        "selection": "frozen hit classifier p>=0.70, >=8 selected hits/2 strings",
        "all": summarize(error, r68, r95),
        "by_hits": [],
        "by_predicted_r68_decile": [],
    }
    hits = predictions.n_hits.to_numpy(dtype=np.int32)
    for low, high in ((8, 12), (12, 16), (16, 24), (24, 40), (40, 1000000)):
        keep = (hits >= low) & (hits < high)
        report["by_hits"].append(
            {
                "hits_low": low,
                "hits_high_exclusive": high,
                **summarize(error[keep], r68[keep], r95[keep]),
            }
        )
    rank = np.argsort(r68)
    for part in np.array_split(rank, 10):
        report["by_predicted_r68_decile"].append(summarize(error[part], r68[part], r95[part]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
