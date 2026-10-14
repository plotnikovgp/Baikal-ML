#!/usr/bin/env python3
"""Pair photon-gated and nominal predictions with standard reco on exact raw events.

The external ROOT file only exposes the preliminary 10/13 cut. The gate is
evaluated on the *same* p>=0.70, 2-string/8-hit cohort as the nominal model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from analyze_reco_central_gap import paired_quantile_delta_ci, quantiles, root_ten_cut


def direction_error(predictions: pd.DataFrame, truth: np.ndarray) -> np.ndarray:
    vectors = predictions[["direction_x", "direction_y", "direction_z"]].to_numpy()
    available = np.isfinite(vectors).all(axis=1)
    result = np.full(len(vectors), np.nan)
    result[available] = np.rad2deg(
        np.arccos(np.clip(np.sum(vectors[available] * truth[available], axis=1), -1, 1))
    )
    return result


def stage_report(
    selected: np.ndarray,
    reco: np.ndarray,
    nominal: np.ndarray,
    candidate: np.ndarray,
    gated: np.ndarray,
    features: dict[str, np.ndarray],
) -> tuple[dict, list[dict]]:
    if not selected.any():
        raise ValueError("Empty comparison cohort")
    pivot = float(np.median(reco[selected]))
    cases = {
        "reco_good_nn_bad": selected & (reco <= pivot) & (nominal > pivot),
        "reco_bad_nn_good": selected & (reco > pivot) & (nominal <= pivot),
    }
    output = {
        "n": int(selected.sum()),
        "gate_reruns": int((selected & gated).sum()),
        "reco": quantiles(reco[selected]),
        "nominal": quantiles(nominal[selected]),
        "photon_gate": quantiles(candidate[selected]),
        "nominal_minus_reco_q50_ci95": paired_quantile_delta_ci(
            nominal[selected], reco[selected], 0.5, seed=4501
        ),
        "gate_minus_reco_q50_ci95": paired_quantile_delta_ci(
            candidate[selected], reco[selected], 0.5, seed=4502
        ),
        "gate_minus_reco_q68_ci95": paired_quantile_delta_ci(
            candidate[selected], reco[selected], 0.68, seed=4505
        ),
        "gate_minus_nominal_q50_ci95": paired_quantile_delta_ci(
            candidate[selected], nominal[selected], 0.5, seed=4503
        ),
        "gate_minus_nominal_q68_ci95": paired_quantile_delta_ci(
            candidate[selected], nominal[selected], 0.68, seed=4504
        ),
        "pivot_reco_q50_deg": pivot,
        "reco_good_nn_bad": int(cases["reco_good_nn_bad"].sum()),
        "reco_bad_nn_good": int(cases["reco_bad_nn_good"].sum()),
        "pivot_shortfall_nominal": int(
            cases["reco_good_nn_bad"].sum() - cases["reco_bad_nn_good"].sum()
        ),
        "pivot_shortfall_gate": int(
            ((selected & (reco <= pivot) & (candidate > pivot)).sum())
            - ((selected & (reco > pivot) & (candidate <= pivot)).sum())
        ),
        "eventwise_gate_better_fraction_rerun": float(
            np.mean(candidate[selected & gated] < nominal[selected & gated])
        )
        if (selected & gated).any()
        else None,
    }
    for key, mask in cases.items():
        output[key + "_profile"] = {
            "n": int(mask.sum()),
            "median_selected_hits": float(np.median(features["selected_hits"][mask])),
            "median_gt_span_m": float(np.median(features["gt_span_m"][mask])),
            "median_truth_theta_deg": float(np.median(features["truth_theta_deg"][mask])),
            "median_root_hits": float(np.median(features["root_hits"][mask])),
            "median_root_triplets": float(np.median(features["root_triplets"][mask])),
        }
    rows = []
    for name, edges in {
        "selected_hits": [8, 12, 20, 35, np.inf],
        "selected_strings": [2, 3, 4, 6, np.inf],
        "gt_span_m": [0, 150, 300, 450, np.inf],
        "truth_theta_deg": [0, 60, 90, 120, 180.001],
        "root_hits": [8, 12, 20, 35, np.inf],
        "root_triplets": [0, 3, 6, 12, np.inf],
    }.items():
        values = features[name]
        for low, high in zip(edges[:-1], edges[1:]):
            mask = selected & (values >= low) & (values < high)
            if mask.sum() < 30:
                continue
            rows.append(
                {
                    "feature": name,
                    "low_inclusive": low,
                    "high_exclusive": high,
                    "n": int(mask.sum()),
                    "gate_reruns": int((mask & gated).sum()),
                    "reco_q50": float(np.median(reco[mask])),
                    "nominal_q50": float(np.median(nominal[mask])),
                    "gate_q50": float(np.median(candidate[mask])),
                    "reco_q68": float(np.quantile(reco[mask], 0.68)),
                    "nominal_q68": float(np.quantile(nominal[mask], 0.68)),
                    "gate_q68": float(np.quantile(candidate[mask], 0.68)),
                    "pivot_shortfall_nominal": int(
                        (mask & (reco <= pivot)).sum() - (mask & (nominal <= pivot)).sum()
                    ),
                    "pivot_shortfall_gate": int(
                        (mask & (reco <= pivot)).sum() - (mask & (candidate <= pivot)).sum()
                    ),
                }
            )
    return output, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--root-fields", type=Path, required=True)
    parser.add_argument("--fair-source", type=Path, required=True)
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    with np.load(args.paired) as pair:
        ids = pair["event_ids"].astype("U")
        fair_index = pair["fair_index"].astype(np.int64)
        reco = pair["standard_reco_direction_error_deg"].astype(float)
        span = pair["gt_span_m"].astype(float)
        official = pair["official_cut"].astype(bool)
    root = pd.read_csv(args.root_fields).set_index("event_id", verify_integrity=True).reindex(ids)
    preliminary = root_ten_cut(root)
    if preliminary.sum() != 44189:
        raise ValueError("Preliminary ROOT cut population changed")
    nominal_rows = pd.read_csv(args.nominal).set_index("event_id", verify_integrity=True)
    candidate_rows = pd.read_csv(args.candidate).set_index("event_id", verify_integrity=True)
    if not nominal_rows.index.equals(candidate_rows.index):
        raise ValueError("Nominal and candidate event order differs")
    available = np.isin(ids, nominal_rows.index)
    nominal_rows = nominal_rows.reindex(ids)
    candidate_rows = candidate_rows.reindex(ids)
    gated = candidate_rows["selected_threshold"].to_numpy() == 0.5
    unchanged = available & ~gated
    if not np.array_equal(
        nominal_rows["n_hits"].to_numpy()[unchanged],
        candidate_rows["n_hits"].to_numpy()[unchanged],
    ):
        raise ValueError("Ungated events have differing hit counts")
    with h5py.File(args.fair_source) as fair:
        fair_ids = fair["test/ev_ids/data"][:][fair_index].astype("U")
        if not np.array_equal(ids, fair_ids):
            raise ValueError("Truth/event alignment differs")
        prime = fair["test/prime_prty/data"][:][fair_index]
    theta = np.deg2rad(prime[:, 0])
    phi = np.deg2rad(prime[:, 1])
    truth = np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )
    nominal = direction_error(nominal_rows, truth)
    candidate = direction_error(candidate_rows, truth)
    if not np.isfinite(nominal[available]).all() or not np.isfinite(candidate[available]).all():
        raise ValueError("Missing prediction on an accepted event")
    unchanged = available & ~gated
    if np.max(np.abs(nominal[unchanged] - candidate[unchanged])) > 0.01:
        raise ValueError("Ungated predictions changed unexpectedly")
    features = {
        "selected_hits": nominal_rows["n_hits"].to_numpy(dtype=float),
        "selected_strings": nominal_rows["n_strings"].to_numpy(dtype=float),
        "gt_span_m": span,
        "truth_theta_deg": prime[:, 0],
        "root_hits": root["fNHits"].to_numpy(dtype=float),
        "root_triplets": root["fNTriplets"].to_numpy(dtype=float),
    }
    summary = {
        "note": "Descriptive external 2020 reco comparison; preliminary ROOT-observable 10/13 cut, not full original 13/13; some primaries overlap training",
        "sources": {k: str(v) for k, v in vars(args).items() if k != "output_dir"},
        "stages": {},
    }
    strata = []
    for name, mask in {
        "root_10_of_13": preliminary & available,
        "root_10_of_13_and_official": preliminary & official & available,
    }.items():
        result, rows = stage_report(mask, reco, nominal, candidate, gated, features)
        summary["stages"][name] = result
        strata.extend({"stage": name, **row} for row in rows)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(strata).to_csv(args.output_dir / "strata.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
