#!/usr/bin/env python3
"""Evaluate direction CSV predictions on one ordered MC split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def track_validity(path: Path) -> dict[str, bool]:
    result: dict[str, bool] = {}
    with h5py.File(path, "r") as h5:
        for split in ("train", "val", "test"):
            ids = h5[f"{split}/event_ids"][:]
            valid = h5[f"{split}/valid"][:]
            for key, value in zip(ids, valid):
                event_id = key.decode() if isinstance(key, bytes) else str(key)
                if event_id in result and result[event_id] != bool(value):
                    raise ValueError(f"Conflicting validity for {event_id}")
                result[event_id] = bool(value)
    return result


def quantiles(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"count": 0, "q50_deg": None, "q68_deg": None, "q95_deg": None}
    return {
        "count": int(len(values)),
        "q50_deg": float(np.quantile(values, 0.5)),
        "q68_deg": float(np.quantile(values, 0.68)),
        "q95_deg": float(np.quantile(values, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--mc-data", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--p070-anchors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pred = pd.read_csv(args.predictions)
    with h5py.File(args.mc_data, "r") as h5:
        total = len(h5[f"{args.split}/ev_ids/data"])
        if "source_event_index" in pred:
            source_index = pred.source_event_index.to_numpy(dtype=np.int64)
            if (
                (source_index < 0).any()
                or (source_index >= total).any()
                or (np.diff(source_index) <= 0).any()
            ):
                raise ValueError("source_event_index must be unique, increasing and in range")
            ids = h5[f"{args.split}/ev_ids/data"][source_index]
            prime = h5[f"{args.split}/prime_prty/data"][source_index, :2]
        else:
            source_index = np.arange(len(pred), dtype=np.int64)
            ids = h5[f"{args.split}/ev_ids/data"][: len(pred)]
            prime = h5[f"{args.split}/prime_prty/data"][: len(pred), :2]
    ids = np.char.decode(ids.astype("S"), "utf-8")
    if len(pred) != len(ids) or not np.array_equal(pred.event_id.to_numpy(), ids):
        raise ValueError("CSV predictions are not aligned with the MC HDF5 event order")
    theta, phi = np.deg2rad(prime[:, 0]), np.deg2rad(prime[:, 1])
    truth = np.stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)),
        axis=1,
    )
    direction = pred[["direction_x", "direction_y", "direction_z"]].to_numpy()
    dot = np.sum(direction * truth, axis=1)
    angle = np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))
    point_error = None
    with h5py.File(args.p070_anchors, "r") as h5:
        anchor_group = h5[args.split]
        if len(anchor_group["event_ids"]) == total:
            anchor_ids = anchor_group["event_ids"][source_index]
            if not np.array_equal(anchor_ids, ids.astype("S")):
                raise ValueError("Anchor rows disagree with the MC HDF5 event order")
            true_point = anchor_group["anchor_point_m"][source_index]
            valid_point = anchor_group["valid"][source_index].astype(bool)
            predicted_point = pred[["anchor_x_m", "anchor_y_m", "anchor_z_m"]].to_numpy()
            delta = predicted_point - true_point
            transverse = delta - np.sum(delta * truth, axis=1)[:, None] * truth
            point_error = np.linalg.norm(transverse, axis=1)
            point_error[~valid_point] = np.nan
    unique = ~pred.event_id.duplicated().to_numpy()
    event_ids = pred.event_id.to_numpy()[unique]
    angle = angle[unique]
    validity = track_validity(args.p070_anchors)
    known = np.asarray([key in validity for key in event_ids])
    single = np.asarray([validity.get(key, False) for key in event_ids])
    if not known.all():
        raise ValueError(f"No multiplicity label for {int((~known).sum())} MC events")
    report = {
        "csv": str(args.predictions),
        "mc_data": str(args.mc_data),
        "split": args.split,
        "original_rows": len(pred),
        "duplicate_id_rows_removed": int((~unique).sum()),
        "all_unique": quantiles(angle),
        "single_track": quantiles(angle[single]),
        "multi_track_or_invalid": quantiles(angle[~single]),
    }
    if point_error is not None:
        unique_point = point_error[unique]
        report["line_m_valid"] = {
            key.replace("_deg", "_m"): value
            for key, value in quantiles(unique_point[np.isfinite(unique_point)]).items()
        }
    else:
        unique_point = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    arrays = dict(
        event_id=event_ids.astype(str),
        angle_deg=angle.astype(np.float32),
        single_track=single,
        n_hits=pred.n_hits.to_numpy()[unique],
        n_strings=pred.n_strings.to_numpy()[unique],
    )
    if unique_point is not None:
        arrays["line_m"] = unique_point.astype(np.float32)
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
