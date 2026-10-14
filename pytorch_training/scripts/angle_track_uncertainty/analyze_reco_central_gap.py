#!/usr/bin/env python3
"""Explain the NN-versus-standard-reco median gap on exactly paired events.

The ROOT preliminary selection is explicitly the 10/13 observable superset;
the three BRecoMuonBDTEval-derived conditions are not claimed to be applied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def root_ten_cut(fields: pd.DataFrame) -> np.ndarray:
    hits = fields["fNHits"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        time_term = np.log10(fields["fTimeTerm"].to_numpy() / (hits - 5))
        triplet_fraction = fields["fNTriplets"].to_numpy() / hits
    return (
        (hits >= 8)
        & (fields["fNStrings"].to_numpy() >= 2)
        & (fields["fCovMatrStat"].to_numpy() == 3)
        & (fields["fScfMaxTheta"].to_numpy() < 1.7)
        & (fields["fEvCenterZ"].to_numpy() < 220)
        & (fields["fZDist"].to_numpy() > 70)
        & (time_term < 1.1)
        & (triplet_fraction > 0.1)
        & (fields["fThetaErr"].to_numpy() < 2.5)
        & (fields["fNCalls"].to_numpy() < 350)
    )


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "q50": float(np.quantile(values, 0.5)),
        "q68": float(np.quantile(values, 0.68)),
        "q95": float(np.quantile(values, 0.95)),
    }


def paired_quantile_delta_ci(
    nn: np.ndarray, reco: np.ndarray, q: float, *, seed: int, reps: int = 400
) -> list[float]:
    rng = np.random.default_rng(seed)
    differences = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        sample = rng.integers(0, len(nn), len(nn))
        differences[index] = np.quantile(nn[sample], q) - np.quantile(reco[sample], q)
    return np.quantile(differences, [0.025, 0.975]).tolist()


def bin_rows(
    name: str,
    values: np.ndarray,
    bounds: list[float],
    selected: np.ndarray,
    reco: np.ndarray,
    nn: np.ndarray,
    pivot: float,
) -> list[dict]:
    rows = []
    for low, high in zip(bounds[:-1], bounds[1:]):
        use = selected & (values >= low) & (values < high)
        if use.sum() < 30:
            continue
        r, n = reco[use], nn[use]
        rows.append(
            {
                "feature": name,
                "low_inclusive": low,
                "high_exclusive": high,
                "count": int(use.sum()),
                "reco_q50_deg": float(np.median(r)),
                "nn_q50_deg": float(np.median(n)),
                "q50_delta_nn_minus_reco_deg": float(np.median(n) - np.median(r)),
                "reco_q68_deg": float(np.quantile(r, 0.68)),
                "nn_q68_deg": float(np.quantile(n, 0.68)),
                "fraction_nn_better_eventwise": float(np.mean(n < r)),
                "reco_below_pivot_fraction": float(np.mean(r <= pivot)),
                "nn_below_pivot_fraction": float(np.mean(n <= pivot)),
                "pivot_count_deficit_nn": int((r <= pivot).sum() - (n <= pivot).sum()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--root-fields", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gt-predictions", type=Path, required=True)
    parser.add_argument("--predicted-hits", type=Path, required=True)
    parser.add_argument("--fair-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    with np.load(args.paired) as paired:
        ids = np.asarray(paired["event_ids"])
        fair_indices = np.asarray(paired["fair_index"], dtype=np.int64)
        span = np.asarray(paired["gt_span_m"], dtype=np.float64)
        official = np.asarray(paired["official_cut"], dtype=bool)
        reco = np.asarray(paired["standard_reco_direction_error_deg"], dtype=np.float64)
        errors = {
            "v4": np.asarray(paired["v4_direction_error_deg"], dtype=np.float64),
            "v4_photon": np.asarray(paired["v4_photon_direction_error_deg"], dtype=np.float64),
            "paired_threshold": np.asarray(
                paired["short_photon_direction_error_deg"], dtype=np.float64
            ),
        }
    root = pd.read_csv(args.root_fields).set_index("event_id", verify_integrity=True)
    if len(root) != len(ids) or set(root.index) != set(ids):
        raise ValueError("ROOT scalar ID coverage differs from paired events")
    root = root.reindex(ids)
    selected = root_ten_cut(root)
    if selected.sum() != 44_189 or (selected & official).sum() != 43_736:
        raise ValueError("The established 10/13 and official-cut counts changed")

    predictions = pd.read_csv(args.predictions).set_index("event_id", verify_integrity=True)
    if not set(ids).issubset(predictions.index):
        raise ValueError("Missing paired prediction rows")
    predictions = predictions.reindex(ids)
    pred_indices = predictions["source_event_index"].to_numpy(dtype=np.int64)
    nn_hits = predictions["n_hits"].to_numpy(dtype=np.int32)
    nn_strings = predictions["n_strings"].to_numpy(dtype=np.int32)
    with h5py.File(args.predicted_hits) as pred, h5py.File(args.fair_source) as fair:
        group = pred["test"]
        stored_ids = np.asarray(group["ev_ids/data"])[pred_indices]
        fair_ids = np.asarray(fair["test/ev_ids/data"])[fair_indices]
        if not np.array_equal(ids.astype("S"), stored_ids) or not np.array_equal(
            ids.astype("S"), fair_ids
        ):
            raise ValueError("Predicted/fair HDF5 event-ID join failed")
        starts = np.asarray(group["ev_starts/data"], dtype=np.int64)
        labels = np.asarray(group["labels/data"])
        lengths = np.diff(starts)
        if not np.array_equal(lengths[pred_indices], nn_hits):
            raise ValueError("Prediction hit counts differ from HDF5")
        noise_per_event = np.add.reduceat((labels == 0).astype(np.int32), starts[:-1])
        track_per_event = np.add.reduceat((labels > 0).astype(np.int32), starts[:-1])
        cascade_per_event = np.add.reduceat((labels < 0).astype(np.int32), starts[:-1])
        noise_fraction = noise_per_event[pred_indices] / nn_hits
        track_fraction = track_per_event[pred_indices] / nn_hits
        cascade_fraction = cascade_per_event[pred_indices] / nn_hits
        prime = np.asarray(fair["test/prime_prty/data"])[fair_indices]
        reco_length = np.asarray(fair["test/reco_ev/data"])[fair_indices, -1]
        gt_starts = np.asarray(fair["test/ev_starts/data"], dtype=np.int64)
        gt_signal_hits = np.diff(gt_starts)[fair_indices]
    if np.any(noise_fraction + track_fraction + cascade_fraction < 1 - 1e-6):
        raise ValueError("Hit label fractions do not sum to one")

    gt_predictions = pd.read_csv(args.gt_predictions).set_index("event_id", verify_integrity=True)
    if not set(ids).issubset(gt_predictions.index):
        raise ValueError("GT-signal inference is missing paired event IDs")
    gt_predictions = gt_predictions.reindex(ids)
    if not np.array_equal(gt_predictions["n_hits"].to_numpy(), gt_signal_hits):
        raise ValueError("GT-signal inference hit counts differ from fair source")
    theta = np.deg2rad(prime[:, 0])
    phi = np.deg2rad(prime[:, 1])
    truth_vectors = np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )
    inferred = gt_predictions[["direction_x", "direction_y", "direction_z"]].to_numpy()
    inferred /= np.linalg.norm(inferred, axis=1, keepdims=True)
    errors["v4_gt_unmasked"] = np.rad2deg(
        np.arccos(np.clip((inferred * truth_vectors).sum(axis=1), -1.0, 1.0))
    )

    features = {
        "nn_hits": (nn_hits, [8, 10, 13, 17, 25, 40, np.inf]),
        "nn_strings": (nn_strings, [2, 3, 4, 5, 7, 10, np.inf]),
        "root_reco_hits_minus_nn_hits": (
            root["fNHits"].to_numpy() - nn_hits,
            [-np.inf, -10, -4, 0, 1, 5, 11, np.inf],
        ),
        "gt_signal_hits_minus_nn_hits": (
            gt_signal_hits - nn_hits,
            [-np.inf, 0, 1, 4, 8, 16, np.inf],
        ),
        "selected_noise_fraction": (noise_fraction, [0, 1e-9, 0.1, 0.25, 0.5, 1.01]),
        "selected_track_fraction": (track_fraction, [0, 1e-9, 0.25, 0.5, 0.75, 1.01]),
        "gt_signal_span_m": (span, [0, 150, 250, 350, 450, np.inf]),
        "true_theta_deg": (prime[:, 0], [0, 30, 60, 90, 120, 150, 181]),
        "log10_true_energy": (
            np.log10(np.maximum(prime[:, 2], 1e-20)),
            [-np.inf, 3, 4, 5, 6, np.inf],
        ),
        "reco_theta_err_deg": (root["fThetaErr"].to_numpy(), [0, 0.2, 0.4, 0.7, 1.2, 2.5]),
        "reco_track_length_m": (reco_length, [-np.inf, 100, 200, 300, 400, 600, np.inf]),
    }
    all_results = {}
    all_strata = []
    for stage_name, mask in (
        ("root_10_of_13", selected),
        ("root_10_of_13_and_official", selected & official),
        ("root_10_of_13_zero_selected_noise", selected & (noise_fraction == 0)),
        (
            "root_10_of_13_zero_noise_more_gt_signal_hits",
            selected & (noise_fraction == 0) & (gt_signal_hits > nn_hits),
        ),
    ):
        stage = {"count": int(mask.sum()), "standard_reco": quantiles(reco[mask])}
        for model_index, (model_name, nn) in enumerate(errors.items()):
            pivot = float(np.median(reco[mask]))
            nn_better = nn[mask] < reco[mask]
            discordant = mask & (reco <= pivot) & (nn > pivot)
            reverse = mask & (reco > pivot) & (nn <= pivot)
            stage[model_name] = {
                **quantiles(nn[mask]),
                "q50_delta_nn_minus_reco_deg": float(np.median(nn[mask]) - pivot),
                "q50_delta_ci95_deg": paired_quantile_delta_ci(
                    nn[mask], reco[mask], 0.5, seed=20261014 + model_index
                ),
                "q68_delta_ci95_deg": paired_quantile_delta_ci(
                    nn[mask], reco[mask], 0.68, seed=20261114 + model_index
                ),
                "fraction_nn_better_eventwise": float(nn_better.mean()),
                "median_paired_delta_deg": float(np.median(nn[mask] - reco[mask])),
                "reco_good_nn_bad_at_pivot": int(discordant.sum()),
                "reco_bad_nn_good_at_pivot": int(reverse.sum()),
                "net_nn_shortfall_at_reco_median": int(discordant.sum() - reverse.sum()),
            }
            if model_name == "v4_photon":
                for feature_name, (values, bounds) in features.items():
                    all_strata.extend(
                        {
                            "stage": stage_name,
                            "model": model_name,
                            **row,
                        }
                        for row in bin_rows(feature_name, values, bounds, mask, reco, nn, pivot)
                    )
                stage[model_name]["discordant_feature_medians"] = {
                    name: float(np.median(values[discordant]))
                    for name, (values, _) in features.items()
                }
                stage[model_name]["reverse_feature_medians"] = {
                    name: float(np.median(values[reverse]))
                    for name, (values, _) in features.items()
                }
        all_results[stage_name] = stage
    summary = {
        "selection_note": "ROOT-available preliminary 10/13 is a superset, not the full 13/13 cut",
        "data_note": "same exact paired nue2 events; p>=0.70 selected NN hits versus standard reco",
        "sources": {key: str(value) for key, value in vars(args).items() if key != "output_dir"},
        "results": all_results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(all_strata).to_csv(args.output_dir / "strata.csv", index=False)
    print(json.dumps({"results": all_results, "strata_rows": len(all_strata)}, indent=2))


if __name__ == "__main__":
    main()
