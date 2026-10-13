#!/usr/bin/env python3
"""Score time-centering input ablations on one fixed muatm/experiment cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from analyze_domain_predictions import matching_weights
from scipy.stats import wasserstein_distance


def read(path: Path) -> pd.DataFrame:
    return (
        pd.read_csv(path)
        .drop_duplicates("event_id", keep="first")
        .set_index("event_id", verify_integrity=True)
    )


def q68(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.68))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--mc-data", type=Path, required=True)
    parser.add_argument("--p070-anchors", type=Path, required=True)
    args = parser.parse_args()
    baseline_exp = read(args.sweep_dir / "exp_a000.csv")
    baseline_mc = read(args.sweep_dir / "muatm_a000.csv")
    index = baseline_mc.source_event_index.to_numpy(dtype=np.int64)
    with h5py.File(args.mc_data, "r") as h5:
        source_ids = np.asarray(h5["val/ev_ids/data"])[index]
        prime = np.asarray(h5["val/prime_prty/data"])[index, :2]
    if not np.array_equal(source_ids.astype("S"), baseline_mc.index.to_numpy(dtype="S")):
        raise ValueError("MC predictions and truth rows do not align")
    theta, phi = np.deg2rad(prime[:, 0]), np.deg2rad(prime[:, 1])
    truth = np.stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)),
        axis=1,
    )
    with h5py.File(args.p070_anchors, "r") as h5:
        single = np.asarray(h5["val/valid"])[index].astype(bool)
        true_point = np.asarray(h5["val/anchor_point_m"])[index]
    weights, support = matching_weights(baseline_exp.reset_index(), baseline_mc.reset_index())
    baseline_exp_direction = baseline_exp[["direction_x", "direction_y", "direction_z"]].to_numpy()
    results = {
        "events": {
            "exp": len(baseline_exp),
            "mc": len(baseline_mc),
            "single_mc": int(single.sum()),
        },
        "matching": support,
        "alpha": {},
    }
    for tag, alpha in (("000", 0.0), ("050", 0.5), ("100", 1.0)):
        exp = read(args.sweep_dir / f"exp_a{tag}.csv").reindex(baseline_exp.index)
        mc = read(args.sweep_dir / f"muatm_a{tag}.csv").reindex(baseline_mc.index)
        if exp.isna().any().any() or mc.isna().any().any():
            raise ValueError("Time-centering variants have different event IDs")
        for old, new in ((baseline_exp, exp), (baseline_mc, mc)):
            if not np.array_equal(old[["n_hits", "n_strings"]], new[["n_hits", "n_strings"]]):
                raise ValueError("Input cut changed across variants")
        direction = mc[["direction_x", "direction_y", "direction_z"]].to_numpy()
        angle = np.rad2deg(np.arccos(np.clip(np.sum(direction * truth, axis=1), -1.0, 1.0)))
        point = mc[["anchor_x_m", "anchor_y_m", "anchor_z_m"]].to_numpy()
        delta = point - true_point
        transverse = delta - np.sum(delta * truth, axis=1)[:, None] * truth
        line = np.linalg.norm(transverse, axis=1)
        exp_direction = exp[["direction_x", "direction_y", "direction_z"]].to_numpy()
        exp_change = np.rad2deg(
            np.arccos(np.clip(np.sum(exp_direction * baseline_exp_direction, axis=1), -1.0, 1.0))
        )
        results["alpha"][tag] = {
            "center_time_alpha": alpha,
            "muatm_mc_angle_q50_q68_deg": np.quantile(angle, [0.5, 0.68]).tolist(),
            "single_muatm_mc_angle_q68_deg": q68(angle[single]),
            "single_muatm_mc_line_q68_m": q68(line[single]),
            "matched_theta_w1_deg": float(
                wasserstein_distance(exp.theta_deg, mc.theta_deg, v_weights=weights)
            ),
            "experimental_direction_change_q50_q68_deg": np.quantile(
                exp_change, [0.5, 0.68]
            ).tolist(),
        }
    path = args.sweep_dir / "summary.json"
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
