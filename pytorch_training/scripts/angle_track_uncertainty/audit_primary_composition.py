#!/usr/bin/env python3
"""Check whether 2020 primary-disjoint nue2 errors reflect sample composition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from audit_primary_overlap import primary_keys


def quantiles(values: np.ndarray) -> list[float]:
    return np.quantile(values, [0.1, 0.5, 0.68, 0.9]).tolist()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(np.interp(q * cumulative[-1], cumulative, sorted_values))


def matched_error(
    error: np.ndarray, novel: np.ndarray, covariates: list[np.ndarray], bins: list[np.ndarray]
) -> dict:
    cells = np.zeros(len(error), dtype=np.int64)
    multiplier = 1
    for values, edges in zip(covariates, bins):
        index = np.digitize(values, edges[1:-1], right=False)
        cells += multiplier * index
        multiplier *= len(edges) - 1
    unique, inverse = np.unique(cells, return_inverse=True)
    novel_counts = np.bincount(inverse[novel], minlength=len(unique))
    reused_counts = np.bincount(inverse[~novel], minlength=len(unique))
    supported = (novel_counts > 0) & (reused_counts >= 5)
    novel_supported = novel & supported[inverse]
    reused_supported = ~novel & supported[inverse]
    weights = novel_counts[inverse[reused_supported]] / reused_counts[inverse[reused_supported]]
    if not novel_supported.any() or not reused_supported.any():
        raise ValueError("No common covariate support")
    return {
        "cells": int(supported.sum()),
        "novel_supported": int(novel_supported.sum()),
        "novel_coverage": float(novel_supported.sum() / novel.sum()),
        "reused_supported": int(reused_supported.sum()),
        "reused_effective_sample_size": float(weights.sum() ** 2 / np.sum(weights**2)),
        "maximum_reused_event_weight": float(weights.max()),
        "novel_q50_q68_deg": np.quantile(error[novel_supported], [0.5, 0.68]).tolist(),
        "reused_matched_q50_q68_deg": [
            weighted_quantile(error[reused_supported], weights, q) for q in (0.5, 0.68)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--pred-p070", type=Path, required=True)
    parser.add_argument("--pred-p050", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--gt-train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    predictions = pd.read_csv(args.predictions)
    with h5py.File(args.pred_p070) as h5:
        ids = np.asarray(h5["test/ev_ids/data"]).astype(str)
        predicted_train = primary_keys(np.asarray(h5["train/prime_prty/data"]))
    if not np.array_equal(ids, predictions.event_id.to_numpy(dtype=str)):
        raise ValueError("p>=0.70 source and prediction rows differ")
    with np.load(args.masks) as masks:
        keep = np.asarray(masks["pred_test"], dtype=bool)
    predictions = predictions.loc[keep].drop_duplicates("event_id", keep="first")
    indices = predictions.source_event_index.to_numpy(dtype=np.int64)
    with h5py.File(args.pred_p050) as h5:
        truth_ids = np.asarray(h5["test/ev_ids/data"])[indices].astype(str)
        truth = np.asarray(h5["test/prime_prty/data"])[indices]
    if not np.array_equal(truth_ids, predictions.event_id.to_numpy(dtype=str)):
        raise ValueError("p>=0.50 truth lookup differs from predictions")
    with h5py.File(args.gt_train) as h5:
        gt_train = primary_keys(np.asarray(h5["train/prime_prty/data"]))
    train_primaries = np.union1d(predicted_train, gt_train)
    novel = ~np.isin(primary_keys(truth), train_primaries)
    nue2 = predictions.event_id.str.startswith("nue2_").to_numpy()
    predictions = predictions.loc[nue2]
    truth = truth[nue2]
    novel = novel[nue2]
    theta, phi = np.deg2rad(truth[:, 0]), np.deg2rad(truth[:, 1])
    direction = np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )
    predicted = predictions[["direction_x", "direction_y", "direction_z"]].to_numpy()
    predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
    error = np.rad2deg(np.arccos(np.clip((direction * predicted).sum(axis=1), -1, 1)))
    hits = predictions.n_hits.to_numpy()
    true_theta = truth[:, 0]
    log_energy = np.log10(np.maximum(truth[:, 2], 1e-6))
    result = {"cohort": "pred_test mask, unique event ID, nue2", "groups": {}, "matching": {}}
    for name, selected in (("primary_disjoint", novel), ("primary_reused", ~novel)):
        result["groups"][name] = {
            "n": int(selected.sum()),
            "angle_q10_q50_q68_q90_deg": quantiles(error[selected]),
            "hits_q10_q50_q68_q90": quantiles(hits[selected]),
            "true_theta_q10_q50_q68_q90_deg": quantiles(true_theta[selected]),
            "log10_energy_q10_q50_q68_q90": quantiles(log_energy[selected]),
        }
    schemes = {
        "hits_energy": ([hits, log_energy], [np.array([0, 12, 16, 24, 40, np.inf]), None]),
        "hits_energy_theta": (
            [hits, log_energy, true_theta],
            [np.array([0, 12, 16, 24, 40, np.inf]), None, np.array([0, 30, 60, 90, 180])],
        ),
    }
    energy_edges = np.quantile(log_energy, [0, 0.2, 0.4, 0.6, 0.8, 1])
    energy_edges[0], energy_edges[-1] = -np.inf, np.inf
    for name, (values, edges) in schemes.items():
        edges[1] = energy_edges
        result["matching"][name] = matched_error(error, novel, values, edges)
        result["matching"][name]["log10_energy_internal_edges"] = energy_edges[1:-1].tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
