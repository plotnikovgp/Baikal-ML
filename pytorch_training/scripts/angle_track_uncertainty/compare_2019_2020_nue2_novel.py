#!/usr/bin/env python3
"""Compare 2019 nue2 holdout F with primary-disjoint 2020 nue2 by topology."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from audit_primary_composition import matched_error, quantiles
from audit_primary_overlap import primary_keys


def angular_error(frame: pd.DataFrame, prime: np.ndarray) -> np.ndarray:
    theta, phi = np.deg2rad(prime[:, 0]), np.deg2rad(prime[:, 1])
    truth = np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )
    prediction = frame[["direction_x", "direction_y", "direction_z"]].to_numpy(copy=True)
    prediction /= np.linalg.norm(prediction, axis=1, keepdims=True)
    return np.rad2deg(np.arccos(np.clip((truth * prediction).sum(axis=1), -1, 1)))


def load_2020(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(args.predictions_2020)
    with h5py.File(args.pred_p070, "r") as h5:
        ids = np.asarray(h5["test/ev_ids/data"]).astype(str)
        train_keys = primary_keys(np.asarray(h5["train/prime_prty/data"]))
    if not np.array_equal(ids, frame.event_id.to_numpy(dtype=str)):
        raise ValueError("2020 prediction/source ID mismatch")
    with np.load(args.masks) as masks:
        keep = np.asarray(masks["pred_test"], dtype=bool)
    frame = frame.loc[keep].drop_duplicates("event_id", keep="first")
    indices = frame.source_event_index.to_numpy(dtype=np.int64)
    with h5py.File(args.pred_p050, "r") as h5:
        truth_ids = np.asarray(h5["test/ev_ids/data"])[indices].astype(str)
        truth = np.asarray(h5["test/prime_prty/data"])[indices]
    if not np.array_equal(truth_ids, frame.event_id.to_numpy(dtype=str)):
        raise ValueError("2020 truth/source ID mismatch")
    with h5py.File(args.gt_train, "r") as h5:
        gt_train_keys = primary_keys(np.asarray(h5["train/prime_prty/data"]))
    train_keys = np.union1d(train_keys, gt_train_keys)
    novel = ~np.isin(primary_keys(truth), train_keys)
    select = novel & frame.event_id.str.startswith("nue2_").to_numpy()
    return frame.loc[select].reset_index(drop=True), truth[select]


def load_2019(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(args.predictions_2019)
    with np.load(args.truth_2019) as archive:
        ids = archive["event_ids"].astype(str)
        prime = archive["prime"]
    index = pd.Index(ids).get_indexer(frame.event_id.astype(str))
    if (index < 0).any() or not np.array_equal(ids[index], frame.event_id.to_numpy(dtype=str)):
        raise ValueError("2019 prediction/truth ID mismatch")
    return frame, prime[index]


def selected_hit_features(frame: pd.DataFrame, path: Path) -> dict[str, np.ndarray]:
    """Reproduce each CSV's eventwise threshold on the p>=0.50 hit file."""
    with h5py.File(path, "r") as h5:
        ids = np.asarray(h5["test/ev_ids/data"]).astype(str)
        starts = np.asarray(h5["test/ev_starts/data"], dtype=np.int64)
        scores = np.asarray(h5["test/signal_probability/data"], dtype=np.float32)
        labels = np.asarray(h5["test/labels/data"], dtype=np.int32)
        residuals_ns = np.asarray(h5["test/t_res/data"], dtype=np.float32)
        data = np.asarray(h5["test/data/data"], dtype=np.float32)
        mean = np.asarray(h5["norm_param/mean"], dtype=np.float32)
        std = np.asarray(h5["norm_param/std"], dtype=np.float32)
    indices = frame.source_event_index.to_numpy(dtype=np.int64)
    if not np.array_equal(ids[indices], frame.event_id.to_numpy(dtype=str)):
        raise ValueError(f"Event ID mismatch for {path}")
    result = {
        key: np.empty(len(frame), dtype=np.float32)
        for key in (
            "z_span_m",
            "noise_fraction",
            "negative_label_fraction",
            "positive_label_fraction",
            "abs_tres_median_ns",
        )
    }
    for row, (index, threshold, expected_hits) in enumerate(
        zip(indices, frame.selected_threshold, frame.n_hits)
    ):
        lo, hi = starts[index : index + 2]
        kept = scores[lo:hi] >= threshold - 1e-6
        if int(kept.sum()) != int(expected_hits):
            raise ValueError(f"Hit count mismatch for {path}, row {row}")
        selected_labels = labels[lo:hi][kept]
        selected_z_m = data[lo:hi, 4][kept] * std[4] + mean[4]
        selected_residuals = residuals_ns[lo:hi][kept]
        result["z_span_m"][row] = np.ptp(selected_z_m)
        result["noise_fraction"][row] = np.mean(selected_labels == 0)
        result["negative_label_fraction"][row] = np.mean(selected_labels < 0)
        result["positive_label_fraction"][row] = np.mean(selected_labels > 0)
        result["abs_tres_median_ns"][row] = np.median(np.abs(selected_residuals))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-2020", type=Path, required=True)
    parser.add_argument("--pred-p070", type=Path, required=True)
    parser.add_argument("--pred-p050", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--gt-train", type=Path, required=True)
    parser.add_argument("--predictions-2019", type=Path, required=True)
    parser.add_argument("--truth-2019", type=Path, required=True)
    parser.add_argument("--pred-p050-2019", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    frame_2020, prime_2020 = load_2020(args)
    frame_2019, prime_2019 = load_2019(args)
    hit_features_2020 = selected_hit_features(frame_2020, args.pred_p050)
    hit_features_2019 = selected_hit_features(frame_2019, args.pred_p050_2019)
    error_2020 = angular_error(frame_2020, prime_2020)
    error_2019 = angular_error(frame_2019, prime_2019)
    error = np.concatenate((error_2020, error_2019))
    reference = np.arange(len(error)) < len(error_2020)
    hits = np.concatenate((frame_2020.n_hits, frame_2019.n_hits))
    strings = np.concatenate((frame_2020.n_strings, frame_2019.n_strings))
    energy = np.log10(np.maximum(np.concatenate((prime_2020[:, 2], prime_2019[:, 2])), 1e-6))
    theta = np.concatenate((prime_2020[:, 0], prime_2019[:, 0]))
    hit_features = {
        key: np.concatenate((hit_features_2020[key], hit_features_2019[key]))
        for key in hit_features_2020
    }
    energy_edges = np.quantile(energy, [0, 0.2, 0.4, 0.6, 0.8, 1])
    energy_edges[0], energy_edges[-1] = -np.inf, np.inf
    hit_edges = np.array([0, 10, 12, 16, 24, 40, np.inf])
    string_edges = np.array([0, 3, 4, 6, 9, np.inf])
    theta_edges = np.array([0, 30, 60, 90, 120, 180])
    z_span_edges = np.array([0, 100, 200, 300, 500, np.inf])
    noise_edges = np.array([0, 0.001, 0.10, 0.20, 0.40, 1.01])
    negative_edges = np.array([0, 0.001, 0.25, 0.50, 0.75, 1.01])
    schemes = {
        "hits": ([hits], [hit_edges]),
        "hits_energy": ([hits, energy], [hit_edges, energy_edges]),
        "hits_energy_theta": ([hits, energy, theta], [hit_edges, energy_edges, theta_edges]),
        "hits_energy_theta_strings": (
            [hits, energy, theta, strings],
            [hit_edges, energy_edges, theta_edges, string_edges],
        ),
        "hits_strings_zspan": (
            [hits, strings, hit_features["z_span_m"]],
            [hit_edges, string_edges, z_span_edges],
        ),
        "hits_energy_theta_zspan": (
            [hits, energy, theta, hit_features["z_span_m"]],
            [hit_edges, energy_edges, theta_edges, z_span_edges],
        ),
        "hits_noise": (
            [hits, hit_features["noise_fraction"]],
            [hit_edges, noise_edges],
        ),
        "hits_noise_negative": (
            [hits, hit_features["noise_fraction"], hit_features["negative_label_fraction"]],
            [hit_edges, noise_edges, negative_edges],
        ),
    }
    result = {
        "reference": "primary-disjoint 2020 nue2, ID-deduplicated pred_test mask, old 50 ns gate",
        "comparison": "2019 F nue2, old 50 ns gate",
        "groups": {},
        "matching": {},
        "log10_energy_internal_edges": energy_edges[1:-1].tolist(),
    }
    for name, selected in (("2020_novel", reference), ("2019_F", ~reference)):
        selected_hits = hits[selected]
        selected_noise = hit_features["noise_fraction"][selected]
        signal_hits = selected_hits - np.rint(selected_hits * selected_noise).astype(int)
        result["groups"][name] = {
            "n": int(selected.sum()),
            "events_with_label_zero_hit": int((selected_noise > 0).sum()),
            "events_below_8_hits_after_oracle_zero_removal": int((signal_hits < 8).sum()),
            "angle_q10_q50_q68_q90_deg": quantiles(error[selected]),
            "hits_q10_q50_q68_q90": quantiles(hits[selected]),
            "strings_q10_q50_q68_q90": quantiles(strings[selected]),
            "log10_energy_q10_q50_q68_q90": quantiles(energy[selected]),
            "true_theta_q10_q50_q68_q90_deg": quantiles(theta[selected]),
            "z_span_q10_q50_q68_q90_m": quantiles(hit_features["z_span_m"][selected]),
            "noise_fraction_q10_q50_q68_q90": quantiles(hit_features["noise_fraction"][selected]),
            "negative_label_fraction_q10_q50_q68_q90": quantiles(
                hit_features["negative_label_fraction"][selected]
            ),
            "positive_label_fraction_q10_q50_q68_q90": quantiles(
                hit_features["positive_label_fraction"][selected]
            ),
            "abs_tres_median_q10_q50_q68_q90_ns": quantiles(
                hit_features["abs_tres_median_ns"][selected]
            ),
        }
    for name, (covariates, bins) in schemes.items():
        match = matched_error(error, reference, covariates, bins)
        result["matching"][name] = {
            "cells": match["cells"],
            "reference_supported": match["novel_supported"],
            "reference_coverage": match["novel_coverage"],
            "comparison_supported": match["reused_supported"],
            "comparison_effective_sample_size": match["reused_effective_sample_size"],
            "maximum_comparison_event_weight": match["maximum_reused_event_weight"],
            "reference_q50_q68_deg": match["novel_q50_q68_deg"],
            "comparison_matched_q50_q68_deg": match["reused_matched_q50_q68_deg"],
        }
    result["2020_novel_slices"] = {}
    for name, selected in (
        ("no_label_zero_hit", hit_features_2020["noise_fraction"] == 0),
        ("some_label_zero_hit", hit_features_2020["noise_fraction"] > 0),
        ("negative_label_majority", hit_features_2020["negative_label_fraction"] > 0.5),
        ("positive_label_majority", hit_features_2020["positive_label_fraction"] > 0.5),
    ):
        result["2020_novel_slices"][name] = {
            "n": int(selected.sum()),
            "angle_q10_q50_q68_q90_deg": quantiles(error_2020[selected]),
            "median_hits": float(np.median(frame_2020.n_hits.to_numpy()[selected])),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
