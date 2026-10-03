#!/usr/bin/env python3
"""Build unambiguous truth-line anchors for an all-particle compact angle set.

The raw MC particle point is converted to the same cluster-local coordinate frame
as the hit positions.  Since any point along an infinite track describes the same
line, the stored anchor is the truth-line point closest to the retained-hit centroid.
Events with more than one simulated track have no unique line target and are
marked invalid for point/uncertainty training while remaining usable for angle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

SPLITS = ("train", "val", "test")
PARTICLES = ("muatm", "nuatm", "nue2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-year", type=int, default=2020)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-events", type=int, default=100_000)
    parser.add_argument("--max-events-per-split", type=int, default=None)
    return parser.parse_args()


def direction_from_angles(prime: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(prime[:, 0].astype(np.float64))
    phi = np.deg2rad(prime[:, 1].astype(np.float64))
    return np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )


def read_indexed(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    """Read arbitrary HDF5 rows while satisfying its sorted-index requirement."""
    unique, inverse = np.unique(indices, return_inverse=True)
    return np.asarray(dataset[unique])[inverse]


def raw_particle_points(
    source: h5py.File,
    particle: str,
    event_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve ``particle_part_event`` IDs directly in the original part layout."""
    count = len(event_ids)
    # muatm part IDs can exceed four digits (e.g. 38186).
    source_parts = np.empty(count, dtype="S16")
    source_indices = np.full(count, -1, dtype=np.int32)
    parsed = np.zeros(count, dtype=bool)
    expected_prefix = particle.split("_")[0].encode()
    for index, event_id in enumerate(event_ids):
        pieces = bytes(event_id).split(b"_")
        if len(pieces) != 3 or pieces[0] != expected_prefix:
            continue
        source_parts[index] = pieces[1]
        parsed[index] = True

    root = source[particle]
    cluster_centers = np.asarray(root["clusters_centers/data"], dtype=np.float64)
    coordinates_are_centered = bool(root["coords_are_cluster_centered/data"][()])
    points = np.full((count, 3), np.nan, dtype=np.float32)
    single_particle = np.zeros(count, dtype=bool)
    order = np.argsort(source_parts, kind="stable")
    ordered_parts = source_parts[order]
    boundaries = np.flatnonzero(ordered_parts[1:] != ordered_parts[:-1]) + 1
    for selected in np.split(order, boundaries):
        selected = selected[parsed[selected]]
        if not len(selected):
            continue
        part = source_parts[selected[0]].decode()
        part_name = f"part_{part}"
        stored_ids = np.asarray(root[f"ev_ids/{part_name}/data"])
        stored_order = np.argsort(stored_ids, kind="stable")
        sorted_stored_ids = stored_ids[stored_order]
        requested_ids = event_ids[selected]
        positions = np.searchsorted(sorted_stored_ids, requested_ids)
        safe_positions = np.minimum(positions, len(sorted_stored_ids) - 1)
        exact = (positions < len(sorted_stored_ids)) & (
            sorted_stored_ids[safe_positions] == requested_ids
        )
        selected = selected[exact]
        event_index = stored_order[safe_positions[exact]].astype(np.int32)
        if not len(selected):
            continue
        source_indices[selected] = event_index

        starts = np.asarray(root[f"muons_prty/mu_starts/{part_name}/data"], dtype=np.int64)
        mu_start = starts[event_index]
        mu_end = starts[event_index + 1]
        one = (mu_end - mu_start) == 1
        single_particle[selected] = one
        nonempty = mu_end > mu_start
        selected = selected[nonempty]
        event_index = event_index[nonempty]
        if not len(selected):
            continue
        individuals = read_indexed(
            root[f"muons_prty/individ/{part_name}/data"], mu_start[nonempty]
        ).astype(np.float64)
        cluster_ids = read_indexed(root[f"raw/cluster_ids/{part_name}/data"], event_index).astype(
            np.int64
        )
        xyz = individuals[:, 2:5]
        if coordinates_are_centered:
            xyz = xyz - cluster_centers[cluster_ids]
        points[selected] = xyz.astype(np.float32)
    return points, parsed, single_particle, source_indices


def event_centers_and_directions(
    compact: h5py.File,
    split: str,
    chunk_events: int,
    max_events: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    group = compact[split]
    starts = group["ev_starts/data"]
    count = len(starts) - 1
    if max_events is not None:
        count = min(count, max_events)
    mean = np.asarray(compact["norm_param/mean"], dtype=np.float64)
    std = np.asarray(compact["norm_param/std"], dtype=np.float64)
    centers = np.empty((count, 3), dtype=np.float32)
    directions = np.empty((count, 3), dtype=np.float32)
    for begin in range(0, count, chunk_events):
        end = min(begin + chunk_events, count)
        event_starts = np.asarray(starts[begin : end + 1], dtype=np.int64)
        hit_begin, hit_end = int(event_starts[0]), int(event_starts[-1])
        relative_starts = event_starts - hit_begin
        lengths = np.diff(relative_starts)
        normalized = np.asarray(group["data/data"][hit_begin:hit_end, 2:5], dtype=np.float64)
        raw = normalized * std[2:5] + mean[2:5]
        sums = np.add.reduceat(raw, relative_starts[:-1], axis=0)
        centers[begin:end] = (sums / lengths[:, None]).astype(np.float32)
        prime = np.asarray(group["prime_prty/data"][begin:end], dtype=np.float32)
        directions[begin:end] = direction_from_angles(prime).astype(np.float32)
        print(f"{split}: centers {end:,}/{count:,}", flush=True)
    return centers, directions


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    with h5py.File(args.compact, "r") as compact, h5py.File(args.source, "r") as source:
        with h5py.File(output, "w") as target:
            target.attrs["compact"] = str(Path(args.compact).resolve())
            target.attrs["source"] = str(Path(args.source).resolve())
            target.attrs["source_particles"] = json.dumps(
                [f"{particle}_{args.source_year}" for particle in PARTICLES]
            )
            target.attrs["definition"] = (
                "single-particle truth-line point closest to the centroid of retained hits"
            )
            for split in SPLITS:
                current_ids = np.asarray(compact[f"{split}/ev_ids/data"])
                if args.max_events_per_split is not None:
                    current_ids = current_ids[: args.max_events_per_split]
                prefixes = current_ids.astype("S6")
                raw_points = np.full((len(current_ids), 3), np.nan, dtype=np.float32)
                parsed = np.zeros(len(current_ids), dtype=bool)
                single_particle = np.zeros(len(current_ids), dtype=bool)
                source_event_indices = np.full(len(current_ids), -1, dtype=np.int32)
                particle_summary = {}
                for particle in PARTICLES:
                    selected = np.flatnonzero(
                        np.char.startswith(prefixes, particle.encode() + b"_")
                    )
                    if not len(selected):
                        continue
                    points, found, single, indices = raw_particle_points(
                        source, f"{particle}_{args.source_year}", current_ids[selected]
                    )
                    raw_points[selected] = points
                    parsed[selected] = found
                    single_particle[selected] = single
                    source_event_indices[selected] = indices
                    particle_summary[particle] = {
                        "events": int(len(selected)),
                        "source_id_parsed": int(found.sum()),
                        "single_track": int(single.sum()),
                    }
                valid = parsed & single_particle & np.isfinite(raw_points).all(axis=1)
                hit_centers, directions = event_centers_and_directions(
                    compact, split, args.chunk_events, args.max_events_per_split
                )
                delta = hit_centers.astype(np.float64) - raw_points.astype(np.float64)
                along = np.sum(delta * directions, axis=1, keepdims=True)
                anchors = raw_points.astype(np.float64) + along * directions
                offsets = anchors - hit_centers
                anchors[~valid] = np.nan
                offsets[~valid] = np.nan

                group = target.create_group(split)
                group.create_dataset("event_ids", data=current_ids, compression="lzf")
                group.create_dataset("valid", data=valid, compression="lzf")
                group.create_dataset(
                    "source_event_index", data=source_event_indices, compression="lzf"
                )
                group.create_dataset("raw_particle_point_m", data=raw_points, compression="lzf")
                group.create_dataset("hit_centroid_m", data=hit_centers, compression="lzf")
                group.create_dataset(
                    "anchor_point_m", data=anchors.astype(np.float32), compression="lzf"
                )
                group.create_dataset(
                    "anchor_offset_m",
                    data=offsets.astype(np.float32),
                    compression="lzf",
                )

                norms = np.linalg.norm(offsets[valid], axis=1)
                split_summary = {
                    "events": int(len(current_ids)),
                    "parsed": int(parsed.sum()),
                    "valid_single_particle": int(valid.sum()),
                    "valid_fraction": float(valid.mean()),
                    "particles": particle_summary,
                    "anchor_offset_norm_quantiles_m": {
                        str(q): float(np.quantile(norms, q))
                        for q in (0.0, 0.5, 0.68, 0.9, 0.95, 0.99, 1.0)
                    },
                }
                summary[split] = split_summary
                print(split, json.dumps(split_summary, sort_keys=True), flush=True)

            target.attrs["summary_json"] = json.dumps(summary, sort_keys=True)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
