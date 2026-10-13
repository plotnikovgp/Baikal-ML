#!/usr/bin/env python3
"""Measure 2019 nue2 transfer and standardize the 2019/2020 span mix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.distance import pdist

SPAN_BINS = np.asarray([0, 100, 150, 200, 250, 300, 400, np.inf])
THETA_BINS = np.asarray([0, 30, 60, 90.0001])


def vectors(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    theta, phi = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    return np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )


def errors(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    prediction = prediction / np.linalg.norm(prediction, axis=1, keepdims=True)
    return np.rad2deg(np.arccos(np.clip(np.sum(prediction * truth, axis=1), -1.0, 1.0)))


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {f"q{int(q * 100)}": float(np.quantile(values, q)) for q in (0.5, 0.68, 0.95)}


def read_csv(path: Path, ids: np.ndarray) -> np.ndarray:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(ids) or any(
        row["event_id"] != event_id.decode() for row, event_id in zip(rows, ids)
    ):
        raise ValueError(f"Prediction IDs/order mismatch: {path}")
    return np.asarray(
        [[float(row[f"direction_{axis}"]) for axis in "xyz"] for row in rows],
        dtype=np.float64,
    )


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    return float(np.interp(q * weights.sum(), np.cumsum(weights[order]), values[order]))


def standardizing_weights(
    source_span: np.ndarray,
    source_theta: np.ndarray,
    target_span: np.ndarray,
    target_theta: np.ndarray,
    *,
    match_theta: bool,
) -> tuple[np.ndarray, float]:
    source_b = np.digitize(source_span, SPAN_BINS) - 1
    target_b = np.digitize(target_span, SPAN_BINS) - 1
    source_t = np.digitize(source_theta, THETA_BINS) - 1
    target_t = np.digitize(target_theta, THETA_BINS) - 1
    weights = np.zeros(len(source_span), dtype=np.float64)
    for span_bin in range(len(SPAN_BINS) - 1):
        theta_range = range(len(THETA_BINS) - 1) if match_theta else range(1)
        for theta_bin in theta_range:
            source_mask = source_b == span_bin
            target_mask = target_b == span_bin
            if match_theta:
                source_mask &= source_t == theta_bin
                target_mask &= target_t == theta_bin
            source_fraction = source_mask.mean()
            target_fraction = target_mask.mean()
            if source_fraction and target_fraction:
                weights[source_mask] = target_fraction / source_fraction
    return weights, float(weights.mean())


def paired_bootstrap(
    current: np.ndarray, baseline: np.ndarray, parts: np.ndarray, seed: int = 2026
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    event_deltas = []
    for _ in range(1000):
        index = rng.integers(0, len(current), len(current))
        event_deltas.append(np.quantile(current[index], 0.68) - np.quantile(baseline[index], 0.68))
    groups = [np.flatnonzero(parts == part) for part in np.unique(parts)]
    part_deltas = []
    for _ in range(1000):
        sampled = rng.integers(0, len(groups), len(groups))
        index = np.concatenate([groups[j] for j in sampled])
        part_deltas.append(np.quantile(current[index], 0.68) - np.quantile(baseline[index], 0.68))
    return {
        "event_95pct": np.quantile(event_deltas, [0.025, 0.975]).tolist(),
        "source_part_95pct": np.quantile(part_deltas, [0.025, 0.975]).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-2019", type=Path, required=True)
    parser.add_argument("--predicted-2019", type=Path, required=True)
    parser.add_argument("--v4-csv", type=Path, required=True)
    parser.add_argument("--photon-csv", type=Path, required=True)
    parser.add_argument("--reco-2020-fair", type=Path, required=True)
    parser.add_argument("--reco-2020-paired", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with h5py.File(args.raw_2019) as raw, h5py.File(args.predicted_2019) as pred:
        raw_group, pred_group = raw["test"], pred["test"]
        raw_ids = raw_group["ev_ids/data"][:]
        pred_ids = pred_group["ev_ids/data"][:]
        lookup = {event_id: index for index, event_id in enumerate(raw_ids)}
        if len(lookup) != len(raw_ids) or any(event_id not in lookup for event_id in pred_ids):
            raise ValueError("Missing or duplicate raw 2019 event IDs")
        starts = raw_group["ev_starts/data"][:]
        features = raw_group["data/data"][:]
        labels = raw_group["labels/data"][:]
        channels = raw_group["channels/data"][:]
        spans = np.empty(len(pred_ids), dtype=np.float64)
        gt_cut = np.empty(len(pred_ids), dtype=bool)
        for j, event_id in enumerate(pred_ids):
            index = lookup[event_id]
            low, high = starts[index : index + 2]
            signal = labels[low:high] != 0
            xyz = features[low:high, 2:5][signal]
            spans[j] = float(np.max(pdist(xyz))) if len(xyz) > 1 else 0.0
            gt_cut[j] = signal.sum() >= 8 and len(np.unique(channels[low:high][signal] // 36)) >= 2
        prime = pred_group["prime_prty/data"][:]
        truth = vectors(prime[:, 0], prime[:, 1])
        v4_error = errors(read_csv(args.v4_csv, pred_ids), truth)
        photon_error = errors(read_csv(args.photon_csv, pred_ids), truth)
        parts = np.asarray([event_id.decode().rsplit("_", 1)[0] for event_id in pred_ids])

    with np.load(args.reco_2020_paired) as paired, h5py.File(args.reco_2020_fair) as fair:
        errors_2020 = np.asarray(paired["short_photon_direction_error_deg"])
        spans_2020 = np.asarray(paired["gt_span_m"])
        theta_2020 = np.asarray(fair["test/prime_prty/data"][:, 0])[paired["fair_index"]]

    clean_error = photon_error[gt_cut]
    clean_span = spans[gt_cut]
    clean_theta = prime[:, 0][gt_cut]
    span_weights, span_coverage = standardizing_weights(
        clean_span, clean_theta, spans_2020, theta_2020, match_theta=False
    )
    joint_weights, joint_coverage = standardizing_weights(
        clean_span, clean_theta, spans_2020, theta_2020, match_theta=True
    )
    bins = []
    for j in range(len(SPAN_BINS) - 1):
        m19 = (clean_span >= SPAN_BINS[j]) & (clean_span < SPAN_BINS[j + 1])
        m20 = (spans_2020 >= SPAN_BINS[j]) & (spans_2020 < SPAN_BINS[j + 1])
        bins.append(
            {
                "span_low_m": float(SPAN_BINS[j]),
                "span_high_m": None
                if not np.isfinite(SPAN_BINS[j + 1])
                else float(SPAN_BINS[j + 1]),
                "n_2019": int(m19.sum()),
                "n_2020": int(m20.sum()),
                "q68_2019_deg": float(np.quantile(clean_error[m19], 0.68)),
                "q68_2020_deg": float(np.quantile(errors_2020[m20], 0.68)),
            }
        )
    summary = {
        "inputs": {key: str(value) for key, value in vars(args).items() if key != "output"},
        "selection": "frozen hit classifier p>=0.70, >=8 selected hits/2 strings; 2019 fair comparison additionally requires GT signal 2s8h",
        "raw_2019_events": int(len(raw_ids)),
        "predicted_2019_events": int(len(pred_ids)),
        "gt_signal_2s8h_2019_events": int(gt_cut.sum()),
        "v4_2019_all": quantiles(v4_error),
        "photon_2019_all": quantiles(photon_error),
        "v4_2019_gt_cut": quantiles(v4_error[gt_cut]),
        "photon_2019_gt_cut": quantiles(clean_error),
        "photon_2020_reco_paired": quantiles(errors_2020),
        "photon_minus_v4_2019_q68_deg": float(
            np.quantile(photon_error, 0.68) - np.quantile(v4_error, 0.68)
        ),
        "photon_minus_v4_2019_q68_bootstrap": paired_bootstrap(photon_error, v4_error, parts),
        "span_2019_median_m": float(np.median(clean_span)),
        "span_2020_median_m": float(np.median(spans_2020)),
        "span_standardized_2019_q68_deg": weighted_quantile(clean_error, span_weights, 0.68),
        "span_standardized_target_coverage": span_coverage,
        "span_theta_standardized_2019_q68_deg": weighted_quantile(clean_error, joint_weights, 0.68),
        "span_theta_standardized_target_coverage": joint_coverage,
        "span_bins": bins,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
