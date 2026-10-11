#!/usr/bin/env python3
"""Pair domain-shift comparisons over the same p>=0.90 MC/experiment IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_domain_predictions import comparison, matching_weights
from scipy.stats import wasserstein_distance

THRESHOLDS = (("070", 0.70), ("075", 0.75), ("080", 0.80), ("085", 0.85), ("090", 0.90))


def fixed_ids(path: Path, *, deduplicate: bool) -> pd.Index:
    frame = pd.read_csv(path)
    if deduplicate:
        frame = frame.drop_duplicates("event_id", keep="first")
    if frame.event_id.duplicated().any():
        raise ValueError(f"Repeated event IDs in {path}")
    return pd.Index(frame.event_id, name="event_id")


def load_fixed(path: Path, ids: pd.Index, *, deduplicate: bool) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if deduplicate:
        frame = frame.drop_duplicates("event_id", keep="first")
    frame = frame.set_index("event_id", verify_integrity=True)
    if not ids.isin(frame.index).all():
        raise ValueError(f"p>=0.90 IDs missing from {path}")
    return frame.loc[ids].reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args()
    root = args.sweep_dir
    exp_ids = fixed_ids(root / "exp_test_p090.csv", deduplicate=False)
    mc_ids = fixed_ids(root / "muatm_val_p090.csv", deduplicate=True)
    frames = {}
    rows = []
    for tag, threshold in THRESHOLDS:
        exp = load_fixed(root / f"exp_test_p{tag}.csv", exp_ids, deduplicate=False)
        mc = load_fixed(root / f"muatm_val_p{tag}.csv", mc_ids, deduplicate=True)
        weights, support = matching_weights(exp, mc)
        frames[tag] = (exp, mc)
        rows.append(
            {
                "threshold": threshold,
                "exp_common_events": len(exp),
                "mc_common_events": len(mc),
                "theta_w1_unweighted_deg": comparison(exp, mc, None)["theta_wasserstein_deg"],
                "theta_w1_hit_string_matched_deg": comparison(exp, mc, weights)[
                    "theta_wasserstein_deg"
                ],
                "matching_effective_mc_events": support["mc_effective_sample_size"],
                "exp_outside_mc_support_fraction": support["exp_outside_mc_support_fraction"],
            }
        )
    if args.bootstrap:
        rng = np.random.default_rng(61)
        deviations = {tag: [] for tag, _ in THRESHOLDS}
        for _ in range(args.bootstrap):
            exp_sample = rng.integers(len(exp_ids), size=len(exp_ids))
            mc_sample = rng.integers(len(mc_ids), size=len(mc_ids))
            distances = {}
            for tag, _ in THRESHOLDS:
                exp, mc = frames[tag]
                sampled_exp, sampled_mc = exp.iloc[exp_sample], mc.iloc[mc_sample]
                weights, _ = matching_weights(sampled_exp, sampled_mc)
                distances[tag] = wasserstein_distance(
                    sampled_exp.theta_deg.to_numpy(),
                    sampled_mc.theta_deg.to_numpy(),
                    v_weights=weights,
                )
            for tag, _ in THRESHOLDS:
                deviations[tag].append(distances[tag] - distances["070"])
        for row, (tag, _) in zip(rows, THRESHOLDS):
            row["theta_w1_delta_vs_p070_ci95_deg"] = np.quantile(
                deviations[tag], [0.025, 0.975]
            ).tolist()
    (root / "domain_common_ids_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(root / "domain_common_ids_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
