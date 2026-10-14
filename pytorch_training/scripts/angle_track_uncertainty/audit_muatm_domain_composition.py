#!/usr/bin/env python3
"""Audit whether experimental theta comparison uses single- or multi-track MC.

The anchor validity flag requires exactly one generated muon, a parsed source
ID and a finite true point. Source metadata is checked so that on this cohort
validity can be interpreted as the exactly-one-muon condition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from match_domain_covariates import join_features, measure


def summary(exp: pd.DataFrame, mc: pd.DataFrame) -> dict:
    theta = mc.theta_deg.to_numpy()
    report = {
        "mc_events": int(len(mc)),
        "mc_theta_q10_q50_q90_deg": np.quantile(theta, [0.1, 0.5, 0.9]).tolist(),
        "mc_n_hits_median": float(mc.n_hits.median()),
        "mc_z_std_median_m": float(mc.z_std_m.median()),
        "matching": {},
    }
    for extra in ((), ("z_std_m",)):
        label = "hit_string" + ("_z_spread" if extra else "")
        report["matching"][label] = measure(exp, mc, extra)
    return report


def align_anchor_validity(
    anchor_ids: np.ndarray,
    valid: np.ndarray,
    source_index: np.ndarray,
    domain_ids: pd.Index,
) -> np.ndarray:
    unique_ids, first, inverse = np.unique(anchor_ids, return_index=True, return_inverse=True)
    if not (
        np.array_equal(valid, valid[first[inverse]])
        and np.array_equal(source_index, source_index[first[inverse]])
    ):
        raise ValueError("Repeated anchor event IDs disagree on track validity/source")
    lookup = {value.decode(): int(first[i]) for i, value in enumerate(unique_ids)}
    index = np.array([lookup.get(value, -1) for value in domain_ids], dtype=np.int64)
    if np.any(index < 0) or not all(value.startswith("muatm_") for value in domain_ids):
        raise ValueError("MC domain IDs are missing from muatm anchors")
    return valid[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "exp_features",
        "mc_features",
        "exp_predictions",
        "mc_predictions",
        "anchors",
        "compact",
        "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    exp = join_features(args.exp_features, args.exp_predictions, deduplicate=False)
    mc = join_features(args.mc_features, args.mc_predictions, deduplicate=True)
    with h5py.File(args.anchors) as anchors, h5py.File(args.compact) as compact:
        anchor_ids = np.asarray(anchors["val/event_ids"], dtype="S64")
        compact_ids = np.asarray(compact["val/ev_ids/data"], dtype="S64")
        if not np.array_equal(anchor_ids, compact_ids):
            raise ValueError("Anchor and selected-hit val event IDs differ")
        valid = np.asarray(anchors["val/valid"], dtype=bool)
        source_index = np.asarray(anchors["val/source_event_index"], dtype=np.int64)
        metadata = json.loads(anchors.attrs["summary_json"])["val"]
    if metadata["parsed"] != metadata["events"] or np.any(source_index < 0):
        raise ValueError("Anchor source matching is incomplete")
    mc_valid = align_anchor_validity(anchor_ids, valid, source_index, mc.index)
    expected_single = metadata["particles"]["muatm"]["single_track"]
    total_val_muatm = metadata["particles"]["muatm"]["events"]
    if int(valid[[value.startswith(b"muatm_") for value in anchor_ids]].sum()) != expected_single:
        raise ValueError("Anchor validity does not match exactly-one-muon metadata")
    report = {
        "selection": "p>=0.70 predicted-signal hits, 2s8h; fixed validation/test domain cohorts",
        "definition": "valid anchor = exactly one generated muon with parsed source and finite point",
        "exp_events": int(len(exp)),
        "val_muatm_rows": int(total_val_muatm),
        "val_single_track_muatm_rows": int(expected_single),
        "val_duplicate_event_ids": int(len(anchor_ids) - len(np.unique(anchor_ids))),
        "domain_muatm_events": int(len(mc)),
        "domain_single_track_events": int(mc_valid.sum()),
        "domain_other_muatm_events": int((~mc_valid).sum()),
        "domain_single_fraction": float(mc_valid.mean()),
        "inputs": {name: str(getattr(args, name)) for name in vars(args) if name != "output"},
        "all_muatm": summary(exp, mc),
        "single_track_muatm": summary(exp, mc.iloc[np.flatnonzero(mc_valid)]),
        "other_muatm": summary(exp, mc.iloc[np.flatnonzero(~mc_valid)]),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
