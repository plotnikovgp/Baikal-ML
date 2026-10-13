#!/usr/bin/env python3
"""Check an idealized Cherenkov track time model on the leak-free MC validation cohort.

This is a feasibility diagnostic, not a fitted reconstruction algorithm.
The free event-time offset absorbs arbitrary HDF5 time origins.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

C_M_PER_NS = 0.299792458


def direction(theta_phi_deg: np.ndarray) -> np.ndarray:
    theta, phi = np.deg2rad(theta_phi_deg[:2])
    return np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        dtype=np.float64,
    )


def photon_time_residual(
    xyz_m: np.ndarray,
    time_ns: np.ndarray,
    line_point_m: np.ndarray,
    line_direction: np.ndarray,
    refractive_index: float,
) -> np.ndarray:
    delta = xyz_m - line_point_m
    along = delta @ line_direction
    transverse = np.linalg.norm(delta - along[:, None] * line_direction, axis=1)
    model_ns = (
        along + np.sqrt(refractive_index * refractive_index - 1.0) * transverse
    ) / C_M_PER_NS
    residual = time_ns - model_ns
    return residual - np.median(residual)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "q50": float(np.quantile(array, 0.5)),
        "q68": float(np.quantile(array, 0.68)),
        "q90": float(np.quantile(array, 0.9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--max-events", type=int, default=5406)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.masks) as mask_file:
        keep = mask_file["pred_val"].astype(bool)
    by_particle: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with h5py.File(args.data, "r") as h5, h5py.File(args.anchors, "r") as anchor:
        group = h5["val"]
        mean = np.asarray(h5["norm_param/mean"], dtype=np.float64)
        std = np.asarray(h5["norm_param/std"], dtype=np.float64)
        indices = np.flatnonzero(keep)
        count = 0
        for event_index in indices:
            prefix = bytes(group["ev_ids/data"][event_index])[:5]
            particle = {
                b"muatm": "muatm",
                b"nuatm": "nuatm",
                b"nue2_": "nue2",
            }.get(prefix, "unknown")
            valid = bool(anchor["val/valid"][event_index])
            if not valid:
                continue
            a = int(group["ev_starts/data"][event_index])
            b = int(group["ev_starts/data"][event_index + 1])
            hit = np.asarray(group["data/data"][a:b], dtype=np.float64) * std + mean
            if len(hit) < 8:
                continue
            truth_direction = direction(np.asarray(group["prime_prty/data"][event_index]))
            centroid = np.asarray(anchor["val/hit_centroid_m"][event_index], dtype=np.float64)
            offset = np.asarray(anchor["val/anchor_offset_m"][event_index], dtype=np.float64)
            point = centroid + offset
            for index in (1.30, 1.33, 1.36):
                residual = photon_time_residual(
                    hit[:, 2:5], hit[:, 1], point, truth_direction, index
                )
                by_particle[particle][f"n{index:.2f}_event_mad_ns"].append(
                    float(np.median(np.abs(residual)))
                )
                by_particle[particle][f"n{index:.2f}_event_q68_abs_ns"].append(
                    float(np.quantile(np.abs(residual), 0.68))
                )
            reversed_residual = photon_time_residual(
                hit[:, 2:5], hit[:, 1], point, -truth_direction, 1.33
            )
            by_particle[particle]["reverse_n1.33_event_mad_ns"].append(
                float(np.median(np.abs(reversed_residual)))
            )
            by_particle[particle]["hit_count"].append(float(len(hit)))
            count += 1
            if count >= args.max_events:
                break
    report = {
        "c_m_per_ns": C_M_PER_NS,
        "free_event_time_offset": "median residual",
        "cohort": "pred_val leak-free, anchor-valid, no multi-track muatm",
        "total_events": count,
        "particle": {
            particle: {name: summarize(values) for name, values in features.items()}
            for particle, features in by_particle.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
