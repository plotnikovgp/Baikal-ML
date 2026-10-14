#!/usr/bin/env python3
"""Check whether multi-muatm MC has a coherent direction target.

Run ``export`` on the compact-data host to make a small exact-ID manifest,
then ``analyze`` on the host with the original raw merged HDF5. The two
stages assert matching event IDs, raw row indices, and primary angles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def direction(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def quantiles(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {}
    return {
        label: float(np.quantile(values, q))
        for label, q in (("q50", 0.5), ("q68", 0.68), ("q90", 0.9))
    }


def export_manifest(anchor_path: Path, compact_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    all_ids, all_indexes, all_primes, all_valid = [], [], [], []
    with h5py.File(anchor_path) as anchors, h5py.File(compact_path) as compact:
        for split in ("val", "test"):
            ids = np.asarray(anchors[f"{split}/event_ids"], dtype="S64")
            source_ids = np.asarray(compact[f"{split}/ev_ids/data"], dtype="S64")
            if not np.array_equal(ids, source_ids):
                raise ValueError(f"Anchor and compact IDs differ in {split}")
            keep = np.fromiter((value.startswith(b"muatm_") for value in ids), bool)
            all_ids.append(ids[keep])
            all_indexes.append(
                np.asarray(anchors[f"{split}/source_event_index"], dtype=np.int32)[keep]
            )
            all_primes.append(
                np.asarray(compact[f"{split}/prime_prty/data"], dtype=np.float32)[keep, :3]
            )
            all_valid.append(np.asarray(anchors[f"{split}/valid"], dtype=bool)[keep])
    ids = np.concatenate(all_ids)
    indexes = np.concatenate(all_indexes)
    primes = np.concatenate(all_primes)
    valid = np.concatenate(all_valid)
    if np.any(indexes < 0):
        raise ValueError("Unresolved raw row index")
    order = np.argsort(ids, kind="stable")
    ids, indexes, primes, valid = ids[order], indexes[order], primes[order], valid[order]
    unique_ids, first, inverse = np.unique(ids, return_index=True, return_inverse=True)
    if not (
        np.array_equal(indexes, indexes[first[inverse]])
        and np.array_equal(primes, primes[first[inverse]])
        and np.array_equal(valid, valid[first[inverse]])
    ):
        raise ValueError("Repeated MC IDs differ in raw source or truth")
    parts = np.asarray([value.split(b"_")[1] for value in unique_ids], dtype="S16")
    np.savez_compressed(
        output,
        event_ids=unique_ids,
        part=parts,
        source_index=indexes[first],
        prime=primes[first],
        anchor_valid=valid[first],
    )
    print(
        json.dumps(
            {
                "rows_before_dedup": len(ids),
                "unique_events": len(unique_ids),
                "parts": int(len(np.unique(parts))),
                "anchor_valid": int(valid[first].sum()),
                "manifest": str(output),
            },
            indent=2,
        ),
        flush=True,
    )


def summarize_group(rows: list[dict], mask: np.ndarray) -> dict:
    selected = [row for row, keep in zip(rows, mask) if keep]
    if not selected:
        return {"n": 0}
    result = {"n": len(selected)}
    for key in (
        "nearest_muon_to_primary_deg",
        "median_muon_to_primary_deg",
        "leading_muon_to_primary_deg",
        "weighted_axis_to_primary_deg",
        "weighted_axis_spread_q68_deg",
        "max_muon_to_primary_deg",
        "transverse_start_spread_q68_m",
    ):
        result[key] = quantiles(np.array([row[key] for row in selected]))
    axis_error = np.array([row["weighted_axis_to_primary_deg"] for row in selected])
    result["axis_within_1deg"] = float(np.mean(axis_error < 1))
    result["axis_within_3deg"] = float(np.mean(axis_error < 3))
    result["axis_within_5deg"] = float(np.mean(axis_error < 5))
    result["all_muons_within_0p01deg"] = float(
        np.mean([row["max_muon_to_primary_deg"] < 0.01 for row in selected])
    )
    result["max_individual_muon_to_primary_deg"] = float(
        max(row["max_muon_to_primary_deg"] for row in selected)
    )
    return result


def analyze(manifest_path: Path, raw_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    with np.load(manifest_path) as manifest:
        event_ids = manifest["event_ids"]
        part = manifest["part"]
        source_index = manifest["source_index"]
        compact_prime = manifest["prime"]
        anchor_valid = manifest["anchor_valid"]
    rows = []
    with h5py.File(raw_path, "r") as source:
        root = source["muatm_2020"]
        for part_id in np.unique(part):
            selected = np.flatnonzero(part == part_id)
            part_name = "part_" + part_id.decode()
            starts = np.asarray(root[f"muons_prty/mu_starts/{part_name}/data"], dtype=np.int64)
            muons = np.asarray(root[f"muons_prty/individ/{part_name}/data"], dtype=np.float64)
            prime = np.asarray(root[f"prime_prty/{part_name}/data"], dtype=np.float64)
            raw_ids = np.asarray(root[f"ev_ids/{part_name}/data"], dtype="S64")
            index = source_index[selected]
            if np.any(index >= len(raw_ids)) or not np.array_equal(
                raw_ids[index], event_ids[selected]
            ):
                raise ValueError(f"Raw event ID/index mismatch in {part_name}")
            angles_agree = np.allclose(
                prime[index, :2], compact_prime[selected, :2], rtol=0, atol=1e-4
            )
            energies_agree = np.allclose(
                prime[index, 2], compact_prime[selected, 2], rtol=1e-6, atol=1e-3
            )
            if not (angles_agree and energies_agree):
                raise ValueError(f"Raw and compact primary truth differ in {part_name}")
            n_mu = starts[index + 1] - starts[index]
            if not np.array_equal(n_mu == 1, anchor_valid[selected]):
                raise ValueError(f"Anchor validity and muon multiplicity differ in {part_name}")
            if np.any((muons[:, 0] < 0) | (muons[:, 0] > np.pi + 1e-5)):
                raise ValueError(f"Muon theta not in radians in {part_name}")
            mu_dir = direction(muons[:, 0], muons[:, 1])
            p = direction(np.deg2rad(prime[index, 0]), np.deg2rad(prime[index, 1]))
            for local, row_index in enumerate(selected):
                first, end = starts[index[local]], starts[index[local] + 1]
                count = int(end - first)
                if not count:
                    rows.append({"n_muons": 0})
                    continue
                unit = mu_dir[first:end]
                angles = angle_deg(unit, p[local])
                energy = muons[first:end, 6]
                good_energy = np.isfinite(energy) & (energy > 0)
                weight = np.where(good_energy, energy, 0)
                if weight.sum() == 0:
                    weight = np.ones(count)
                axis = np.average(unit, axis=0, weights=weight)
                norm = np.linalg.norm(axis)
                axis = axis / norm if norm > 1e-12 else unit[np.argmax(weight)]
                spread = angle_deg(unit, axis)
                positions = muons[first:end, 2:5]
                transverse = np.cross(positions - positions.mean(axis=0), p[local])
                transverse_distance = np.linalg.norm(transverse, axis=1)
                rows.append(
                    {
                        "n_muons": count,
                        "nearest_muon_to_primary_deg": float(angles.min()),
                        "median_muon_to_primary_deg": float(np.median(angles)),
                        "leading_muon_to_primary_deg": float(angles[np.argmax(weight)]),
                        "weighted_axis_to_primary_deg": float(angle_deg(axis, p[local])),
                        "weighted_axis_spread_q68_deg": float(np.quantile(spread, 0.68)),
                        "max_muon_to_primary_deg": float(angles.max()),
                        "transverse_start_spread_q68_m": float(
                            np.quantile(transverse_distance, 0.68)
                        ),
                    }
                )
            print(f"{part_name}: {len(selected)} selected events", flush=True)
    multiplicity = np.array([row["n_muons"] for row in rows])
    if len(rows) != len(event_ids):
        raise ValueError("Event count changed during raw audit")
    report = {
        "manifest": str(manifest_path),
        "raw": str(raw_path),
        "selected_unique_muatm_events": int(len(rows)),
        "parts": int(len(np.unique(part))),
        "muon_multiplicity": {
            label: int(mask.sum())
            for label, mask in (
                ("zero", multiplicity == 0),
                ("one", multiplicity == 1),
                ("two_to_four", (multiplicity >= 2) & (multiplicity <= 4)),
                ("five_to_nine", (multiplicity >= 5) & (multiplicity <= 9)),
                ("ten_plus", multiplicity >= 10),
            )
        },
        "groups": {
            label: summarize_group(rows, mask)
            for label, mask in (
                ("one", multiplicity == 1),
                ("two_to_four", (multiplicity >= 2) & (multiplicity <= 4)),
                ("five_to_nine", (multiplicity >= 5) & (multiplicity <= 9)),
                ("ten_plus", multiplicity >= 10),
                ("multi_all", multiplicity >= 2),
            )
        },
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    export = sub.add_parser("export")
    export.add_argument("--anchors", type=Path, required=True)
    export.add_argument("--compact", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    audit = sub.add_parser("analyze")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--raw", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "export":
        export_manifest(args.anchors, args.compact, args.output)
    else:
        analyze(args.manifest, args.raw, args.output)


if __name__ == "__main__":
    main()
