#!/usr/bin/env python3
"""Diagnose paired reco/NN central errors against full-raw MC hit truth.

The ROOT cuts available here are only the observable preliminary 10/13.
This is a diagnostic of the 2020 external reco cohort, not an independent
model-selection validation set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from analyze_reco_central_gap import root_ten_cut


def event_counts(group: h5py.Group) -> dict[str, np.ndarray]:
    starts = group["ev_starts/data"][:].astype(np.int64)
    lengths = np.diff(starts)
    if np.any(lengths <= 0):
        raise ValueError("Empty events are not supported")
    label = group["labels/data"][:]
    if len(label) != starts[-1]:
        raise ValueError("Label and event-start lengths differ")
    counts = {"hits": lengths}
    for name, mask in {
        "track": label > 0,
        "cascade": label < 0,
        "noise": label == 0,
    }.items():
        counts[name] = np.add.reduceat(mask.astype(np.int32), starts[:-1])
    return counts


def direction_error(frame: pd.DataFrame, truth: np.ndarray) -> np.ndarray:
    vectors = frame[["direction_x", "direction_y", "direction_z"]].to_numpy(float)
    norms = np.linalg.norm(vectors, axis=1)
    out = np.full(len(frame), np.nan)
    valid = np.isfinite(vectors).all(axis=1) & (norms > 0)
    out[valid] = np.rad2deg(
        np.arccos(
            np.clip(np.sum(vectors[valid] / norms[valid, None] * truth[valid], axis=1), -1, 1)
        )
    )
    return out


def angle_components(
    theta: np.ndarray, phi: np.ndarray, truth_angles: np.ndarray, mask: np.ndarray
) -> dict:
    theta_error = np.abs(theta[mask] - truth_angles[mask, 0])
    phi_error = np.abs((phi[mask] - truth_angles[mask, 1] + 180) % 360 - 180)
    return {
        "theta_q50": float(np.median(theta_error)),
        "theta_q68": float(np.quantile(theta_error, 0.68)),
        "phi_q50": float(np.median(phi_error)),
        "phi_q68": float(np.quantile(phi_error, 0.68)),
    }


def describe(
    mask: np.ndarray, reco: np.ndarray, nominal: np.ndarray, gate: np.ndarray, pivot: float
) -> dict:
    if mask.sum() < 30:
        return {}
    return {
        "n": int(mask.sum()),
        "reco_q50": float(np.median(reco[mask])),
        "nominal_q50": float(np.median(nominal[mask])),
        "gate_q50": float(np.median(gate[mask])),
        "reco_q68": float(np.quantile(reco[mask], 0.68)),
        "gate_q68": float(np.quantile(gate[mask], 0.68)),
        "gate_better_event_fraction": float(np.mean(gate[mask] < reco[mask])),
        "gate_minus_reco_paired_median": float(np.median(gate[mask] - reco[mask])),
        "pivot_deficit_gate": int(np.sum(reco[mask] <= pivot) - np.sum(gate[mask] <= pivot)),
    }


def bin_rows(
    base: np.ndarray,
    features: dict[str, tuple[np.ndarray, list[float]]],
    reco: np.ndarray,
    nominal: np.ndarray,
    gate: np.ndarray,
    pivot: float,
) -> list[dict]:
    rows = []
    for name, (values, edges) in features.items():
        for low, high in zip(edges[:-1], edges[1:]):
            mask = base & (values >= low) & (values < high)
            result = describe(mask, reco, nominal, gate, pivot)
            if result:
                rows.append({"feature": name, "low": low, "high": high, **result})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--root-fields", type=Path, required=True)
    parser.add_argument("--fair-source", type=Path, required=True)
    parser.add_argument("--raw-hits", type=Path, required=True)
    parser.add_argument("--selected-hits", type=Path, required=True)
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    with np.load(args.paired) as pair:
        ids = pair["event_ids"].astype("U")
        fair_index = pair["fair_index"].astype(np.int64)
        reco = pair["standard_reco_direction_error_deg"].astype(float)
        gt_span = pair["gt_span_m"].astype(float)
        official = pair["official_cut"].astype(bool)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate paired IDs")
    root = pd.read_csv(args.root_fields).set_index("event_id", verify_integrity=True).reindex(ids)
    preliminary = root_ten_cut(root)
    nominal_df = pd.read_csv(args.nominal).set_index("event_id", verify_integrity=True)
    gate_df = pd.read_csv(args.gate).set_index("event_id", verify_integrity=True)
    if not nominal_df.index.equals(gate_df.index):
        raise ValueError("Prediction IDs/order differ")
    nominal_ids = set(nominal_df.index)
    available = np.fromiter(
        (event_id in nominal_ids for event_id in ids), dtype=bool, count=len(ids)
    )
    nominal_df = nominal_df.reindex(ids)
    gate_df = gate_df.reindex(ids)

    with (
        h5py.File(args.fair_source) as fair,
        h5py.File(args.raw_hits) as raw,
        h5py.File(args.selected_hits) as selected,
    ):
        fair_ids = fair["test/ev_ids/data"][:][fair_index].astype("U")
        if not np.array_equal(ids, fair_ids):
            raise ValueError("Fair source/event IDs differ")
        prime = fair["test/prime_prty/data"][:][fair_index]
        raw_ids = raw["test/ev_ids/data"][:].astype("U")
        if len(np.unique(raw_ids)) != len(raw_ids):
            raise ValueError("Duplicate exact full-raw IDs")
        raw_index = pd.Index(raw_ids).get_indexer(ids)
        if np.any(raw_index < 0):
            raise ValueError("Exact full-raw event ID is missing")
        selected_ids = selected["test/ev_ids/data"][:].astype("U")
        if len(np.unique(selected_ids)) != len(selected_ids):
            raise ValueError("Duplicate selected IDs")
        if set(selected_ids) != nominal_ids:
            raise ValueError("Prediction and selected-hit event sets differ")
        selected_index = pd.Index(selected_ids).get_indexer(ids)
        if not np.array_equal(available, selected_index >= 0):
            raise ValueError("Selected event availability differs")
        raw_counts = {key: values[raw_index] for key, values in event_counts(raw["test"]).items()}
        selected_counts = event_counts(selected["test"])

    theta, phi = np.deg2rad(prime[:, 0]), np.deg2rad(prime[:, 1])
    truth = np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )
    nominal = direction_error(nominal_df, truth)
    gate = direction_error(gate_df, truth)
    if not (np.isfinite(nominal[available]).all() and np.isfinite(gate[available]).all()):
        raise ValueError("Nonfinite available prediction")
    if not np.array_equal(
        nominal_df["n_hits"].to_numpy()[available],
        selected_counts["hits"][selected_index[available]],
    ):
        raise ValueError("Nominal hit count/selected input differs")
    raw_signal = raw_counts["track"] + raw_counts["cascade"]
    selected_track = np.full(len(ids), np.nan)
    selected_cascade = np.full(len(ids), np.nan)
    selected_noise = np.full(len(ids), np.nan)
    selected_hits = np.full(len(ids), np.nan)
    for key, out in (
        ("track", selected_track),
        ("cascade", selected_cascade),
        ("noise", selected_noise),
        ("hits", selected_hits),
    ):
        out[available] = selected_counts[key][selected_index[available]]
    if np.any(selected_track[available] > raw_counts["track"][available]) or np.any(
        selected_cascade[available] > raw_counts["cascade"][available]
    ):
        raise ValueError("Selected signal hit count exceeds raw truth count")
    signal_recall = np.divide(
        selected_track + selected_cascade,
        raw_signal,
        out=np.full(len(ids), np.nan),
        where=raw_signal > 0,
    )
    track_recall = np.divide(
        selected_track,
        raw_counts["track"],
        out=np.full(len(ids), np.nan),
        where=raw_counts["track"] > 0,
    )
    noise_fraction = selected_noise / selected_hits
    reco_hits = root["fNHits"].to_numpy(float)
    reco_triplets = root["fNTriplets"].to_numpy(float)
    base = preliminary & available
    if base.sum() != 42_003:
        raise ValueError(f"Expected 42,003 paired 10/13 events, got {base.sum()}")
    pivot = float(np.median(reco[base]))
    gate_loss = base & (reco <= pivot) & (gate > pivot)
    gate_rescue = base & (reco > pivot) & (gate <= pivot)
    features = {
        "raw_signal_hits": (raw_signal, [0, 12, 20, 35, 60, np.inf]),
        "raw_track_fraction": (
            raw_counts["track"] / np.maximum(raw_signal, 1),
            [0, 0.5, 0.8, 0.95, 1.001],
        ),
        "selected_signal_recall": (signal_recall, [0, 0.5, 0.7, 0.9, 0.99, 1.001]),
        "selected_track_recall": (track_recall, [0, 0.5, 0.7, 0.9, 0.99, 1.001]),
        "selected_noise_fraction": (noise_fraction, [0, 1e-9, 0.1, 0.25, 0.5, 1.001]),
        "root_hits_minus_selected": (
            reco_hits - selected_hits,
            [-np.inf, -10, -4, 0, 1, 5, 11, np.inf],
        ),
        "root_theta_error": (root["fThetaErr"].to_numpy(float), [0, 0.2, 0.4, 0.7, 1.2, 2.5]),
        "root_triplets": (reco_triplets, [0, 3, 6, 12, 20, np.inf]),
        "selected_hits": (selected_hits, [8, 12, 20, 35, np.inf]),
        "gt_span_m": (gt_span, [0, 150, 300, 450, np.inf]),
    }
    profiles = {}
    for name, mask in {"gate_loss": gate_loss, "gate_rescue": gate_rescue}.items():
        profiles[name] = {
            "n": int(mask.sum()),
            "median_selected_hits": float(np.median(selected_hits[mask])),
            "median_raw_signal_hits": float(np.median(raw_signal[mask])),
            "median_signal_recall": float(np.median(signal_recall[mask])),
            "median_track_recall": float(np.nanmedian(track_recall[mask])),
            "zero_selected_noise_fraction": float(np.mean(selected_noise[mask] == 0)),
            "median_root_hits": float(np.median(reco_hits[mask])),
            "median_root_triplets": float(np.median(reco_triplets[mask])),
            "median_root_theta_error": float(np.median(root["fThetaErr"].to_numpy(float)[mask])),
            "median_gt_span_m": float(np.median(gt_span[mask])),
        }
    strata = bin_rows(base, features, reco, nominal, gate, pivot)
    # Paired checks test whether high-reco-quality events remain difficult
    # even when the classifier keeps nearly all true signal hits.
    check_masks = {
        "all": base,
        "high_reco_quality": base & (reco_hits >= 20) & (reco_triplets >= 12),
        "high_reco_quality_recall_ge_90": base
        & (reco_hits >= 20)
        & (reco_triplets >= 12)
        & (signal_recall >= 0.9),
        "high_reco_quality_recall_lt_70": base
        & (reco_hits >= 20)
        & (reco_triplets >= 12)
        & (signal_recall < 0.7),
        "selected_8_to_11_recall_ge_90": base & (selected_hits < 12) & (signal_recall >= 0.9),
        "selected_8_to_11_recall_lt_70": base & (selected_hits < 12) & (signal_recall < 0.7),
        "official_additional": base & official,
    }
    checks = {}
    for name, mask in check_masks.items():
        checks[name] = describe(mask, reco, nominal, gate, pivot)
    gate_vectors = gate_df[["direction_x", "direction_y", "direction_z"]].to_numpy(float)
    gate_theta = np.rad2deg(np.arccos(np.clip(gate_vectors[:, 2], -1, 1)))
    gate_phi = np.rad2deg(np.arctan2(gate_vectors[:, 1], gate_vectors[:, 0]))
    components = {}
    for name in ("all", "high_reco_quality", "high_reco_quality_recall_ge_90"):
        mask = check_masks[name]
        components[name] = {
            "gate": angle_components(gate_theta, gate_phi, prime[:, :2], mask),
            "standard_reco": angle_components(
                root["fThetaRec"].to_numpy(float),
                root["fPhiRec"].to_numpy(float),
                prime[:, :2],
                mask,
            ),
        }
    parts = np.asarray([event_id.split("_")[1] for event_id in ids])
    source_parts = {}
    for part in sorted(set(parts)):
        mask = base & (parts == part)
        if mask.sum() < 30:
            continue
        source_parts[part] = {
            "all": describe(mask, reco, nominal, gate, pivot),
            "high_reco_quality": describe(
                mask & check_masks["high_reco_quality"], reco, nominal, gate, pivot
            ),
        }
    nominal_r68 = nominal_df["angle_r68_deg"].to_numpy(float)
    uncertainty = {}
    for name in ("all", "high_reco_quality", "high_reco_quality_recall_ge_90"):
        mask = check_masks[name]
        if not np.isfinite(nominal_r68[mask]).all():
            raise ValueError("Missing nominal p>=0.70 calibrated radius")
        uncertainty[name] = {
            "median_nominal_r68_deg": float(np.median(nominal_r68[mask])),
            "nominal_empirical_coverage": float(np.mean(nominal[mask] <= nominal_r68[mask])),
        }
    output = {
        "note": "Descriptive external 2020 reco MC; ROOT-observable preliminary 10/13 only; train-primary overlap possible",
        "inputs": {k: str(v) for k, v in vars(args).items() if k != "output_dir"},
        "pivot_reco_q50": pivot,
        "gate_loss_vs_rescue": profiles,
        "checks": checks,
        "components": components,
        "source_parts": source_parts,
        "nominal_uncertainty": uncertainty,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    pd.DataFrame(strata).to_csv(args.output_dir / "strata.csv", index=False)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
