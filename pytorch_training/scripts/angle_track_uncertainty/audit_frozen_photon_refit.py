#!/usr/bin/env python3
"""Evaluate an existing NN-seeded Cherenkov refit on exactly paired events.

The fit prior is fixed from earlier MC validation. The 2020 reco-quality
selection is descriptive and cannot itself be applied to experimental data.
The optional track-only fit uses MC truth labels and is an oracle diagnostic,
never a deployable input mode.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from analyze_reco_central_gap import root_ten_cut
from fit_photon_track import fit_one


def unit(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    theta, phi = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    return np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "q50": float(np.quantile(values, 0.5)),
        "q68": float(np.quantile(values, 0.68)),
        "q90": float(np.quantile(values, 0.9)),
    }


def summarize(frame: pd.DataFrame) -> dict:
    before = frame.before_deg.to_numpy(float)
    after = frame.after_deg.to_numpy(float)
    result = {
        "n": int(len(frame)),
        "nn_before_deg": quantiles(before),
        "refit_deg": quantiles(after),
        "refit_better_fraction": float(np.mean(after < before)),
        "direction_shift_deg": quantiles(frame.direction_shift_deg.to_numpy(float)),
        "fit_success_fraction": float(frame.fit_success.mean()),
        "median_fit_wall_ms": float(frame.fit_wall_ms.median()),
        "median_initial_mad_ns": float(frame.initial_mad_ns.median()),
        "median_final_mad_ns": float(frame.final_mad_ns.median()),
    }
    if "reco_deg" in frame:
        result["standard_reco_deg"] = quantiles(frame.reco_deg.to_numpy(float))
    rng = np.random.default_rng(20261014)
    changes = []
    for _ in range(1000):
        selected = rng.integers(0, len(frame), len(frame))
        changes.append(float(np.median(after[selected]) - np.median(before[selected])))
    result["refit_minus_nn_q50_bootstrap_95"] = np.quantile(changes, [0.025, 0.975]).tolist()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--paired", type=Path)
    parser.add_argument("--root-fields", type=Path)
    parser.add_argument("--fair-source", type=Path)
    parser.add_argument("--min-selected-hits", type=int, default=20)
    parser.add_argument("--fit-hit-label", choices=("all", "track-only"), default="all")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.paired is None) != (args.root_fields is None):
        raise ValueError("Provide paired and root-fields together")
    if (args.paired is None) != (args.fair_source is None):
        raise ValueError("Provide paired and fair-source together")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    predictions = pd.read_csv(args.predictions).set_index("event_id", verify_integrity=True)
    with h5py.File(args.data) as h5:
        split = h5[args.split]
        ids = np.asarray(split["ev_ids/data"]).astype("U")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Repeated input event IDs")
        if set(ids) != set(predictions.index):
            raise ValueError("Input and prediction event sets differ")
        predictions = predictions.reindex(ids)
        starts = np.asarray(split["ev_starts/data"], dtype=np.int64)
        prime_full = np.asarray(split["prime_prty/data"][:, :3])
        prime = np.asarray(prime_full[:, :2], dtype=np.float64)
        stored_mean = np.asarray(h5["norm_param/mean"], dtype=np.float64)
        stored_std = np.asarray(h5["norm_param/std"], dtype=np.float64)
        hits = np.asarray(split["data/data"], dtype=np.float64) * stored_std + stored_mean
        labels = np.asarray(split["labels/data"]) if args.fit_hit_label == "track-only" else None
    lengths = np.diff(starts)
    if not np.array_equal(lengths, predictions.n_hits.to_numpy(int)):
        raise ValueError("HDF5 and prediction hit counts differ")
    keep = (
        np.ones(len(ids), dtype=bool)
        if args.paired is not None
        else lengths >= args.min_selected_hits
    )
    reco_error = None
    if args.paired is not None:
        with np.load(args.paired) as pair:
            pair_ids = pair["event_ids"].astype("U")
            fair_index = np.asarray(pair["fair_index"], dtype=np.int64)
            pair_error = np.asarray(pair["standard_reco_direction_error_deg"], dtype=float)
        if len(np.unique(pair_ids)) != len(pair_ids):
            raise ValueError("Repeated standard-reco event IDs")
        root = pd.read_csv(args.root_fields).set_index("event_id", verify_integrity=True)
        if set(pair_ids) != set(root.index):
            raise ValueError("Paired and ROOT-field event sets differ")
        root = root.reindex(ids)
        pair_index = pd.Index(pair_ids).get_indexer(ids)
        reco_error = np.full(len(ids), np.nan)
        present = pair_index >= 0
        with h5py.File(args.fair_source) as fair:
            fair_ids = fair["test/ev_ids/data"][:][fair_index].astype("U")
            fair_prime = fair["test/prime_prty/data"][:][fair_index, :3]
        if not np.array_equal(pair_ids, fair_ids):
            raise ValueError("Paired and fair-source event IDs differ")
        if not np.array_equal(prime_full[present], fair_prime[pair_index[present]]):
            raise ValueError("Predicted-hit and standard-reco truth triples differ")
        reco_error[present] = pair_error[pair_index[present]]
        keep &= present & root_ten_cut(root)
        keep &= (root.fNHits.to_numpy(float) >= 20) & (root.fNTriplets.to_numpy(float) >= 12)
        if args.max_events is None and int(keep.sum()) != 5679:
            raise ValueError(f"Expected 5,679 exact paired reco events, got {keep.sum()}")
    selected = np.flatnonzero(keep)
    if args.max_events is not None:
        selected = selected[: args.max_events]
    if not len(selected):
        raise ValueError("No selected events")
    pred_vec = predictions[["direction_x", "direction_y", "direction_z"]].to_numpy(float)
    pred_norm = np.linalg.norm(pred_vec, axis=1)
    pred_vec /= pred_norm[:, None]
    anchor = predictions[["anchor_x_m", "anchor_y_m", "anchor_z_m"]].to_numpy(float)
    true_vec = unit(prime[:, 0], prime[:, 1])
    rows = []
    skipped_few_fit_hits = 0
    for number, index in enumerate(selected):
        hit = hits[starts[index] : starts[index + 1]]
        if labels is not None:
            selected_labels = labels[starts[index] : starts[index + 1]]
            hit = hit[selected_labels > 0]
            if len(hit) < 8:
                skipped_few_fit_hits += 1
                continue
        xyz = hit[:, 2:5]
        start = time.monotonic()
        fitted, _point, diagnostic = fit_one(
            xyz,
            hit[:, 1],
            pred_vec[index],
            anchor[index] - xyz.mean(axis=0),
            refractive_index=1.36,
            time_scale_ns=10.0,
            angle_prior_deg=1.0,
            point_prior_m=10.0,
            max_nfev=50,
        )
        row = {
            "event_id": ids[index],
            "n_hits": int(lengths[index]),
            "fit_hits": int(len(hit)),
            "before_deg": float(angle_deg(pred_vec[index], true_vec[index])),
            "after_deg": float(angle_deg(fitted, true_vec[index])),
            "direction_shift_deg": diagnostic["direction_shift_deg"],
            "fit_success": diagnostic["success"],
            "fit_wall_ms": (time.monotonic() - start) * 1000,
            "initial_mad_ns": diagnostic["initial_mad_ns"],
            "final_mad_ns": diagnostic["final_mad_ns"],
        }
        if reco_error is not None:
            row["reco_deg"] = float(reco_error[index])
        rows.append(row)
        if (number + 1) % 500 == 0:
            print(f"fitted={number + 1}/{len(selected)}", flush=True)
    frame = pd.DataFrame(rows)
    report = {
        "inputs": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "selection": (
            "ROOT-observable 10/13 plus fNHits>=20 and fNTriplets>=12"
            if args.paired is not None
            else f"selected p>=0.70 hits>={args.min_selected_hits}"
        ),
        "fit": "existing fit_one; n=1.36, time scale=10ns, angle prior=1deg, point prior=10m",
        "fit_hit_label_mode": args.fit_hit_label,
        "skipped_fewer_than_eight_fit_hits": skipped_few_fit_hits,
        "all": summarize(frame),
    }
    args.output_dir.mkdir(parents=True)
    frame.to_csv(args.output_dir / "events.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
