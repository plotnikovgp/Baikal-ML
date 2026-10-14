#!/usr/bin/env python3
"""Relate a rejected photon-time refit to forbidden-at-inference hit labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {"q50": float(np.quantile(values, 0.5)), "q68": float(np.quantile(values, 0.68))}


def describe(frame: pd.DataFrame) -> dict:
    if not len(frame):
        return {"n": 0}
    return {
        "n": int(len(frame)),
        "nn_deg": quantiles(frame.before_deg.to_numpy(float)),
        "refit_deg": quantiles(frame.after_deg.to_numpy(float)),
        "reco_deg": quantiles(frame.reco_deg.to_numpy(float)),
        "refit_better_fraction": float(np.mean(frame.after_deg < frame.before_deg)),
        "median_track_fraction": float(frame.track_fraction.median()),
        "median_noise_hits": float(frame.noise_hits.median()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--refit-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    events = pd.read_csv(args.refit_events)
    if not events.event_id.is_unique:
        raise ValueError("Repeated refit event IDs")
    with h5py.File(args.data) as h5:
        split = h5["test"]
        ids = np.asarray(split["ev_ids/data"]).astype("U")
        starts = np.asarray(split["ev_starts/data"], dtype=np.int64)
        labels = np.asarray(split["labels/data"])
    if not pd.Index(ids).is_unique:
        raise ValueError("Repeated input event IDs")
    index = pd.Index(ids).get_indexer(events.event_id)
    if np.any(index < 0):
        raise ValueError("Refit events absent from input HDF5")
    track = np.add.reduceat((labels > 0).astype(np.int32), starts[:-1])[index]
    cascade = np.add.reduceat((labels < 0).astype(np.int32), starts[:-1])[index]
    noise = np.add.reduceat((labels == 0).astype(np.int32), starts[:-1])[index]
    counts = np.diff(starts)[index]
    if not np.array_equal(track + cascade + noise, counts):
        raise ValueError("Hit-label count mismatch")
    if not np.array_equal(counts, events.n_hits.to_numpy(int)):
        raise ValueError("Refit/input hit counts differ")
    events["track_hits"] = track
    events["cascade_hits"] = cascade
    events["noise_hits"] = noise
    events["track_fraction"] = track / counts
    report = {
        "selection": "exact paired 2020 reco 10/13, fNHits>=20, fNTriplets>=12, p>=0.70 hits",
        "warning": "MC hit labels are oracle diagnostic features, unavailable for experimental inference",
        "all": describe(events),
        "track_hits_q50_q68": quantiles(track),
        "cascade_hits_q50_q68": quantiles(cascade),
        "noise_hits_q50_q68": quantiles(noise),
        "track_fraction_q50_q68": quantiles(events.track_fraction.to_numpy()),
        "at_least_eight_track_hits_fraction": float(np.mean(track >= 8)),
        "zero_noise_fraction": float(np.mean(noise == 0)),
        "track_fraction_bins": {},
        "noise_groups": {
            "zero": describe(events.loc[noise == 0]),
            "positive": describe(events.loc[noise > 0]),
        },
    }
    edges = [0, 0.5, 0.8, 0.95, 1.001]
    for low, high in zip(edges[:-1], edges[1:]):
        subset = events.loc[(events.track_fraction >= low) & (events.track_fraction < high)]
        report["track_fraction_bins"][f"{low:g}-{high:g}"] = describe(subset)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
