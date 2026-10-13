#!/usr/bin/env python3
"""Paired reco/NN audit of a conditional hit gate on identical event IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def error(frame: pd.DataFrame, truth: np.ndarray) -> np.ndarray:
    theta, phi = np.deg2rad(truth[:, 0]), np.deg2rad(truth[:, 1])
    target = np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )
    pred = frame[["direction_x", "direction_y", "direction_z"]].to_numpy()
    pred /= np.linalg.norm(pred, axis=1, keepdims=True)
    return np.rad2deg(np.arccos(np.clip(np.sum(pred * target, axis=1), -1, 1)))


def quantiles(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, q)) for q in (0.5, 0.68, 0.95)]


def bootstrap(candidate: np.ndarray, baseline: np.ndarray) -> list[float]:
    rng = np.random.default_rng(20261013)
    changes = []
    for _ in range(500):
        index = rng.integers(0, len(candidate), len(candidate))
        changes.append(np.quantile(candidate[index], 0.68) - np.quantile(baseline[index], 0.68))
    return np.quantile(changes, [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--fair", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--root-fields", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.paired) as paired:
        ids = np.asarray(paired["event_ids"]).astype(str)
        fair_index = np.asarray(paired["fair_index"], dtype=np.int64)
        official = np.asarray(paired["official_cut"], dtype=bool)
        standard_error = np.asarray(paired["standard_reco_direction_error_deg"])
        old_error = np.asarray(paired["short_photon_direction_error_deg"])
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Paired event IDs are not unique")
    with h5py.File(args.fair) as h5:
        truth = np.asarray(h5["test/prime_prty/data"])[fair_index]
        true_ids = np.asarray(h5["test/ev_ids/data"])[fair_index].astype(str)
    if not np.array_equal(ids, true_ids):
        raise ValueError("Paired IDs do not match fair HDF5 source")
    base = pd.read_csv(args.baseline).set_index("event_id", verify_integrity=True)
    candidate = pd.read_csv(args.candidate).set_index("event_id", verify_integrity=True)
    if not pd.Index(ids).isin(base.index).all() or not pd.Index(ids).isin(candidate.index).all():
        raise ValueError("Missing paired predictions")
    base = base.reindex(ids)
    candidate = candidate.reindex(ids)
    e_base, e_candidate = error(base, truth), error(candidate, truth)
    if np.max(np.abs(e_base - old_error)) > 0.1:
        raise ValueError("Existing paired direction error disagrees with CSV baseline")
    fields = pd.read_csv(args.root_fields).set_index("event_id", verify_integrity=True)
    fields = fields.reindex(ids)
    if fields.isna().any().any():
        raise ValueError("Missing ROOT fields")
    hits = fields.fNHits.to_numpy()
    strings = fields.fNStrings.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        partial = (
            (hits >= 8)
            & (strings >= 2)
            & (fields.fCovMatrStat.to_numpy() == 3)
            & (fields.fScfMaxTheta.to_numpy() < 1.7)
            & (fields.fEvCenterZ.to_numpy() < 220)
            & (fields.fZDist.to_numpy() > 70)
            & (np.log10(fields.fTimeTerm.to_numpy() / (hits - 5)) < 1.1)
            & (fields.fNTriplets.to_numpy() / hits > 0.1)
            & (fields.fThetaErr.to_numpy() < 2.5)
            & (fields.fNCalls.to_numpy() < 350)
        )
    result = {"paired_event_ids": len(ids), "stages": {}}
    for name, selected in (
        ("all", np.ones(len(ids), dtype=bool)),
        ("official", official),
        ("root_available_10_of_13", partial),
        ("official_and_10_of_13", official & partial),
    ):
        eb, ec, er = e_base[selected], e_candidate[selected], standard_error[selected]
        result["stages"][name] = {
            "n": int(selected.sum()),
            "standard_q50_q68_q95": quantiles(er),
            "baseline_q50_q68_q95": quantiles(eb),
            "candidate_q50_q68_q95": quantiles(ec),
            "candidate_minus_baseline_q68_ci95": bootstrap(ec, eb),
            "candidate_reruns": int(
                (candidate.selected_threshold.to_numpy()[selected] < 0.6).sum()
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
