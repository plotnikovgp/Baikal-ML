#!/usr/bin/env python3
"""Compare the same angle model after hit classification on full raw reco pulses.

Both the old reco-masked and new unmasked branches use the same frozen
signal/noise checkpoint, threshold, 2-string/8-hit rule, and v4 angle bundle.
Only exact event/cluster matches from ``extract_reco_unmasked_exact.py`` are
eligible.  The ROOT cut is the observable preliminary 10/13 superset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from analyze_reco_central_gap import paired_quantile_delta_ci, quantiles, root_ten_cut


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--root-fields", type=Path, required=True)
    parser.add_argument("--fair-source", type=Path, required=True)
    parser.add_argument("--old-predictions", type=Path, required=True)
    parser.add_argument("--raw-selected", type=Path, required=True)
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    with np.load(args.paired) as paired:
        ids = np.asarray(paired["event_ids"])
        fair_indices = np.asarray(paired["fair_index"], dtype=np.int64)
        official = np.asarray(paired["official_cut"], dtype=bool)
        reco = np.asarray(paired["standard_reco_direction_error_deg"], dtype=np.float64)
        old = np.asarray(paired["v4_direction_error_deg"], dtype=np.float64)
        old_photon = np.asarray(paired["v4_photon_direction_error_deg"], dtype=np.float64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Paired IDs are not unique")
    root = pd.read_csv(args.root_fields).set_index("event_id", verify_integrity=True)
    if set(root.index) != set(ids):
        raise ValueError("ROOT field ID coverage differs from paired events")
    preliminary = root_ten_cut(root.reindex(ids))
    if preliminary.sum() != 44_189:
        raise ValueError("Established preliminary 10/13 count changed")
    old_predictions = pd.read_csv(args.old_predictions).set_index("event_id", verify_integrity=True)
    if not set(ids).issubset(old_predictions.index):
        raise ValueError("Old predictions missing paired events")
    old_predictions = old_predictions.reindex(ids)
    raw_predictions = pd.read_csv(args.raw_predictions).set_index("event_id", verify_integrity=True)
    if not set(raw_predictions.index).issubset(set(ids)):
        raise ValueError("Raw predictions contain an event outside the paired set")
    available = np.isin(ids, raw_predictions.index.to_numpy())
    raw_predictions = raw_predictions.reindex(ids)

    with h5py.File(args.fair_source) as fair, h5py.File(args.raw_selected) as selected:
        fair_ids = np.asarray(fair["test/ev_ids/data"])[fair_indices]
        if not np.array_equal(ids.astype("S"), fair_ids):
            raise ValueError("Fair truth and paired IDs differ")
        prime = np.asarray(fair["test/prime_prty/data"])[fair_indices]
        selected_ids = np.asarray(selected["test/ev_ids/data"]).astype("U")
        if not np.array_equal(selected_ids, pd.read_csv(args.raw_predictions)["event_id"]):
            raise ValueError("Raw selected-hit and inference rows are not aligned")
        starts = np.asarray(selected["test/ev_starts/data"], dtype=np.int64)
        labels = np.asarray(selected["test/labels/data"])
        raw_noise = np.add.reduceat((labels == 0).astype(np.int32), starts[:-1])
        selected_lengths = np.diff(starts)
        if not np.array_equal(
            selected_lengths, pd.read_csv(args.raw_predictions)["n_hits"].to_numpy()
        ):
            raise ValueError("Raw selected-hit counts and inference rows differ")
    theta = np.deg2rad(prime[:, 0])
    phi = np.deg2rad(prime[:, 1])
    truth = np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )
    inferred = raw_predictions[["direction_x", "direction_y", "direction_z"]].to_numpy()
    lengths = np.linalg.norm(inferred[available], axis=1)
    if not np.allclose(lengths, 1.0, atol=1e-4):
        raise ValueError("Raw direction vectors are not unit length")
    raw = np.full(len(ids), np.nan, dtype=np.float64)
    raw[available] = np.rad2deg(
        np.arccos(np.clip((inferred[available] * truth[available]).sum(axis=1), -1, 1))
    )
    raw_noise_series = pd.Series(raw_noise / selected_lengths, index=selected_ids)
    noise_fraction = raw_noise_series.reindex(ids).to_numpy()
    raw_hits = raw_predictions["n_hits"].to_numpy(dtype=np.float64)
    old_hits = old_predictions["n_hits"].to_numpy(dtype=np.float64)

    stages = {
        "root_10_of_13": preliminary,
        "root_10_of_13_and_official": preliminary & official,
    }
    results = {}
    strata_rows = []
    for stage_name, stage_mask in stages.items():
        common = stage_mask & available
        lost = stage_mask & ~available
        if not common.any():
            raise ValueError(f"No common events in {stage_name}")
        branch = {
            "root_cut_events": int(stage_mask.sum()),
            "raw_classifier_accepted": int(common.sum()),
            "raw_classifier_rejected": int(lost.sum()),
            "raw_classifier_acceptance": float(common.sum() / stage_mask.sum()),
            "raw_rejected_standard_reco": quantiles(reco[lost]) if lost.any() else None,
            "raw_rejected_old_v4": quantiles(old[lost]) if lost.any() else None,
            "on_common_events": {
                "standard_reco": quantiles(reco[common]),
                "old_v4_reco_masked": quantiles(old[common]),
                "old_v4_photon_reco_masked": quantiles(old_photon[common]),
                "raw_unmasked_v4": quantiles(raw[common]),
                "median_old_selected_hits": float(np.median(old_hits[common])),
                "median_raw_selected_hits": float(np.median(raw_hits[common])),
                "raw_zero_selected_noise_fraction": float(np.mean(noise_fraction[common] == 0)),
                "raw_v4_better_than_reco_eventwise": float(np.mean(raw[common] < reco[common])),
                "raw_v4_better_than_old_v4_eventwise": float(np.mean(raw[common] < old[common])),
                "raw_minus_reco_q50_ci95": paired_quantile_delta_ci(
                    raw[common], reco[common], 0.5, seed=20261015
                ),
                "raw_minus_reco_q68_ci95": paired_quantile_delta_ci(
                    raw[common], reco[common], 0.68, seed=20261016
                ),
                "raw_minus_old_q50_ci95": paired_quantile_delta_ci(
                    raw[common], old[common], 0.5, seed=20261017
                ),
                "raw_minus_old_q68_ci95": paired_quantile_delta_ci(
                    raw[common], old[common], 0.68, seed=20261018
                ),
            },
        }
        pivot = float(np.median(reco[common]))
        branch["on_common_events"]["old_v4_pivot_shortfall"] = int(
            ((reco[common] <= pivot) & (old[common] > pivot)).sum()
            - ((reco[common] > pivot) & (old[common] <= pivot)).sum()
        )
        branch["on_common_events"]["raw_v4_pivot_shortfall"] = int(
            ((reco[common] <= pivot) & (raw[common] > pivot)).sum()
            - ((reco[common] > pivot) & (raw[common] <= pivot)).sum()
        )
        for feature_name, feature, bounds in (
            ("old_strings", old_predictions["n_strings"].to_numpy(), [2, 3, 4, 5, 7, 10]),
            ("raw_strings", raw_predictions["n_strings"].to_numpy(), [2, 3, 4, 5, 7, 10]),
            ("raw_hits", raw_hits, [8, 13, 17, 25, 40, np.inf]),
        ):
            for low, high in zip(bounds[:-1], bounds[1:]):
                use = common & (feature >= low) & (feature < high)
                if use.sum() < 30:
                    continue
                strata_rows.append(
                    {
                        "stage": stage_name,
                        "feature": feature_name,
                        "low_inclusive": low,
                        "high_exclusive": high,
                        "count": int(use.sum()),
                        "standard_reco_q50": float(np.median(reco[use])),
                        "old_v4_q50": float(np.median(old[use])),
                        "raw_v4_q50": float(np.median(raw[use])),
                        "standard_reco_q68": float(np.quantile(reco[use], 0.68)),
                        "old_v4_q68": float(np.quantile(old[use], 0.68)),
                        "raw_v4_q68": float(np.quantile(raw[use], 0.68)),
                    }
                )
        results[stage_name] = branch
    summary = {
        "selection_note": "ROOT-available preliminary 10/13 superset, not exact full 13/13",
        "input_note": "same 57,316 reco event IDs and exact raw clusters; p>=0.70, 2 strings/8 hits; frozen classifier and v4 bundle",
        "sources": {key: str(value) for key, value in vars(args).items() if key != "output_dir"},
        "results": results,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(strata_rows).to_csv(args.output_dir / "strata.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
