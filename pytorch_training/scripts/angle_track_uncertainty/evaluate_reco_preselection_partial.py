#!/usr/bin/env python3
"""Paired reco/NN resolution under the ROOT-available preliminary cuts.

This is a 10-of-13 cut, not the full fillTrackAnalysisTree_v1.0
fPrelCutsPassed condition. Three BRecoMuonBDTEval-derived variables are
not stored in the source reco ROOT/HDF5 files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

MISSING = [
    "fLogPHit_30_min > -9",
    "fQacc_20 / fQtot_20 > 0.05",
    "fTRes_30_med < 60",
]


def q50_q68(values: np.ndarray, selected: np.ndarray) -> dict[str, float]:
    subset = values[selected]
    return {
        "q50_deg": float(np.quantile(subset, 0.5)),
        "q68_deg": float(np.quantile(subset, 0.68)),
    }


def paired_q68_interval(
    candidate: np.ndarray, standard: np.ndarray, selected: np.ndarray, reps: int, seed: int
) -> dict[str, object]:
    candidate, standard = candidate[selected], standard[selected]
    rng = np.random.default_rng(seed)
    differences = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        sampled = rng.integers(0, len(candidate), len(candidate))
        differences[index] = np.quantile(candidate[sampled], 0.68) - np.quantile(
            standard[sampled], 0.68
        )
    return {
        "delta_deg": float(np.quantile(candidate, 0.68) - np.quantile(standard, 0.68)),
        "ci95_deg": np.quantile(differences, [0.025, 0.975]).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-events", type=Path, required=True)
    parser.add_argument("--root-fields", type=Path, required=True)
    parser.add_argument("--fair-source", type=Path, required=True)
    parser.add_argument("--quality-sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.bootstrap < 1:
        raise ValueError("Bootstrap repetitions must be positive")

    paired = np.load(args.paired_events)
    ids = paired["event_ids"]
    source = pd.read_csv(args.root_fields).set_index("event_id", verify_integrity=True)
    if len(source) != len(ids) or set(ids) != set(source.index):
        raise ValueError("ROOT scalars do not cover the exact paired event set")
    source = source.reindex(ids)
    if source.isna().any().any():
        raise ValueError("Missing ROOT scalars after event-ID join")

    with h5py.File(args.fair_source) as fair, h5py.File(args.quality_sidecar) as quality:
        indices = paired["fair_index"]
        h5_ids = np.asarray(fair["test/ev_ids/data"])[indices]
        if not np.array_equal(ids.astype("S"), h5_ids):
            raise ValueError("Paired IDs differ from fair HDF5 indices")
        reco_ev = np.asarray(fair["test/reco_ev/data"])[indices]
        quality_theta = np.asarray(quality["test/theta_deg"])[indices]
        quality_phi = np.asarray(quality["test/phi_deg"])[indices]
    consistency = {}
    for name, column in (
        ("fTimeTerm", 1),
        ("fNHits", 4),
        ("fNStrings", 5),
        ("fThetaRec", 7),
        ("fPhiRec", 8),
    ):
        mismatch = np.trunc(source[name].to_numpy()) != reco_ev[:, column]
        consistency[name] = int(mismatch.sum())
    if any(consistency.values()):
        raise ValueError(f"ROOT/HDF5 event mapping failed: {consistency}")
    for name, sidecar in (("fThetaRec", quality_theta), ("fPhiRec", quality_phi)):
        if not np.allclose(source[name].to_numpy(), sidecar, atol=1e-10, rtol=0):
            raise ValueError(f"ROOT and official-quality sidecar disagree on {name}")

    fields = {name: source[name].to_numpy() for name in source.columns}
    hits = fields["fNHits"]
    strings = fields["fNStrings"]
    with np.errstate(divide="ignore", invalid="ignore"):
        time_quality = np.log10(fields["fTimeTerm"] / (hits - 5))
        triplet_fraction = fields["fNTriplets"] / hits
    cuts = {
        "fNHits >= 8": hits >= 8,
        "fNStrings >= 2": strings >= 2,
        "fCovMatrStat == 3": fields["fCovMatrStat"] == 3,
        "fScfMaxTheta < 1.7": fields["fScfMaxTheta"] < 1.7,
        "fEvCenterZ < 220": fields["fEvCenterZ"] < 220,
        "fZDist > 70": fields["fZDist"] > 70,
        "log10(fTimeTerm/(fNHits-5)) < 1.1": time_quality < 1.1,
        "fNTriplets/fNHits > 0.1": triplet_fraction > 0.1,
        "fThetaErr < 2.5": fields["fThetaErr"] < 2.5,
        "fNCalls < 350": fields["fNCalls"] < 350,
    }
    selected = np.ones(len(ids), dtype=bool)
    cut_counts = []
    for expression, passed in cuts.items():
        selected &= passed
        cut_counts.append(
            {
                "cut": expression,
                "pass_individually": int(passed.sum()),
                "pass_sequentially": int(selected.sum()),
            }
        )
    official = paired["official_cut"].astype(bool)
    errors = {
        "standard_reco": paired["standard_reco_direction_error_deg"],
        "v4_photon": paired["v4_photon_direction_error_deg"],
        "paired_threshold_candidate": paired["short_photon_direction_error_deg"],
    }
    stages = {
        "no_additional_cut": np.ones(len(ids), dtype=bool),
        "official_reco_cut": official,
        "root_available_preliminary_10_of_13": selected,
        "official_and_root_available_preliminary": official & selected,
    }
    results = {}
    for stage_index, (stage, mask) in enumerate(stages.items()):
        row = {"count": int(mask.sum())}
        for name, values in errors.items():
            row[name] = q50_q68(values, mask)
            row[name]["ge20_count"] = int((values[mask] >= 20).sum())
        row["candidate_minus_standard_reco_q68"] = paired_q68_interval(
            errors["paired_threshold_candidate"],
            errors["standard_reco"],
            mask,
            args.bootstrap,
            26 + stage_index,
        )
        results[stage] = row
    output = {
        "selection": "Exact common nue2 reco test; predicted signal p>=0.70 and 8 hits/2 strings at NN input",
        "available_cut_count": len(cuts),
        "full_preliminary_cut_count": len(cuts) + len(MISSING),
        "missing_conditions": MISSING,
        "theta_rec_lt_80": "Disabled in supplied C++ source; not applied",
        "event_id_mapping_checks": consistency,
        "root_sidecar_angles_exact": True,
        "paired_bootstrap_repetitions": args.bootstrap,
        "cut_counts": cut_counts,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
