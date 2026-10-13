#!/usr/bin/env python3
"""Compare fixed-cohort experimental and muatm-MC theta under a hit-time gate.

Input matching weights use only the original p>=0.70 hit/string counts and
vertical hit spread. The candidate theta values cannot change those weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from match_domain_covariates import measure

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_predictions(path: Path, *, muatm_only: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if muatm_only:
        frame = frame[frame.event_id.str.startswith("muatm_")].drop_duplicates(
            "event_id", keep="first"
        )
    return frame.set_index("event_id", verify_integrity=True)


def fixed_z_spread(path: Path, frame: pd.DataFrame) -> np.ndarray:
    with h5py.File(path) as h5:
        group = h5["test"]
        starts = np.asarray(group["ev_starts/data"], dtype=np.int64)
        ids = np.asarray(group["ev_ids/data"])[frame.source_event_index.to_numpy()].astype(str)
        if not np.array_equal(ids, frame.index.to_numpy(dtype=str)):
            raise ValueError(f"Prediction IDs do not match {path}")
        z = np.asarray(group["data/data"][:, 4], dtype=np.float64)
        z = z * float(h5["norm_param/std"][4]) + float(h5["norm_param/mean"][4])
        selected = np.asarray(group["signal_probability/data"][:]) >= 0.7
    count_prefix = np.r_[0, np.cumsum(selected)]
    z_prefix = np.r_[0.0, np.cumsum(np.where(selected, z, 0.0))]
    z2_prefix = np.r_[0.0, np.cumsum(np.where(selected, z * z, 0.0))]
    indices = frame.source_event_index.to_numpy(dtype=np.int64)
    lo, hi = starts[indices], starts[indices + 1]
    count = count_prefix[hi] - count_prefix[lo]
    if not np.array_equal(count, frame.n_hits.to_numpy(dtype=np.int64)):
        raise ValueError("Baseline p>=0.70 hit counts differ from HDF5 selection")
    mean_z = (z_prefix[hi] - z_prefix[lo]) / count
    mean_z2 = (z2_prefix[hi] - z2_prefix[lo]) / count
    return np.sqrt(np.maximum(mean_z2 - mean_z * mean_z, 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-baseline", type=Path, required=True)
    parser.add_argument("--exp-candidate", type=Path, required=True)
    parser.add_argument("--mc-baseline", type=Path, required=True)
    parser.add_argument("--mc-candidate", type=Path, required=True)
    parser.add_argument("--exp-data", type=Path, required=True)
    parser.add_argument("--mc-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError(args.output_dir / "summary.json")
    exp_base = read_predictions(args.exp_baseline)
    exp_candidate = read_predictions(args.exp_candidate)
    mc_base = read_predictions(args.mc_baseline, muatm_only=True)
    mc_candidate = read_predictions(args.mc_candidate, muatm_only=True)
    for name, baseline, candidate in (
        ("experiment", exp_base, exp_candidate),
        ("muatm MC", mc_base, mc_candidate),
    ):
        if not baseline.index.equals(candidate.index):
            raise ValueError(f"{name} candidate cohort/order changed")
    # The baseline CSV may come from a separately prepared p>=0.70 HDF5,
    # whereas the candidate always indexes the p>=0.50 HDF5 given here.
    exp_index = exp_candidate[["source_event_index"]].assign(n_hits=exp_base.n_hits)
    mc_index = mc_candidate[["source_event_index"]].assign(n_hits=mc_base.n_hits)
    exp_z = fixed_z_spread(args.exp_data, exp_index)
    mc_z = fixed_z_spread(args.mc_data, mc_index)
    exp_cov = exp_base[["n_hits", "n_strings"]].copy()
    mc_cov = mc_base[["n_hits", "n_strings"]].copy()
    exp_cov["z_std_m"] = exp_z
    mc_cov["z_std_m"] = mc_z
    summary = {
        "experiment_events": len(exp_base),
        "unique_muatm_mc_events": len(mc_base),
        "matching_covariates": "fixed baseline p>=0.70 selected-hit counts and z spread",
        "candidate_reruns_experiment": int((exp_candidate.selected_threshold < 0.6).sum()),
        "candidate_reruns_muatm": int((mc_candidate.selected_threshold < 0.6).sum()),
        "methods": {},
    }
    curves = []
    for label, exp_pred, mc_pred in (
        ("p>=0.70", exp_base, mc_base),
        ("photon gate 50 ns", exp_candidate, mc_candidate),
    ):
        exp = exp_cov.assign(theta_deg=exp_pred.theta_deg.to_numpy())
        mc = mc_cov.assign(theta_deg=mc_pred.theta_deg.to_numpy())
        hit_string, hit_weights = measure(exp, mc, (), return_weights=True)
        hit_string_z, z_weights = measure(exp, mc, ("z_std_m",), return_weights=True)
        match = {"hit_string": hit_string, "hit_string_z_spread": hit_string_z}
        summary["methods"][label] = {
            "experiment_theta_q10_q50_q90_deg": np.quantile(
                exp.theta_deg, [0.1, 0.5, 0.9]
            ).tolist(),
            "matching": match,
        }
        curves.append(
            (label, exp.theta_deg.to_numpy(), mc.theta_deg.to_numpy(), hit_weights, z_weights)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    bins = np.arange(0, 186, 5)
    for ax, (label, exp_theta, mc_theta, hit_weights, z_weights) in zip(axes, curves):
        ax.hist(
            exp_theta, bins=bins, density=True, histtype="step", linewidth=2, label="Experiment"
        )
        ax.hist(
            mc_theta,
            bins=bins,
            weights=hit_weights,
            density=True,
            histtype="step",
            linewidth=2,
            label="muatm MC, hits/strings matched",
        )
        ax.hist(
            mc_theta,
            bins=bins,
            weights=z_weights,
            density=True,
            histtype="step",
            linewidth=1.5,
            label="muatm MC, plus z spread",
        )
        ax.set(xlabel="Predicted θ [deg]", title=label)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Density / 5° bin")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "theta_distribution.png", dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
