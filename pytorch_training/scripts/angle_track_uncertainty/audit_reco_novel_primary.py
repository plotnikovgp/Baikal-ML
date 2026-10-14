#!/usr/bin/env python3
"""Audit NN versus standard reco on primary-disjoint 2020 reco events.

The input HDF5 must contain p>=0.70 predicted-signal hits with the 2s8h cut.
Use exact float32 (theta, phi, energy) primary fingerprints to exclude every
primary found in internal 2020 train, validation, or test files. Never tune a
model on the output of this audit and call the same cohort an independent test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from audit_primary_overlap import primary_keys


def vectors_from_angles(angles: np.ndarray) -> np.ndarray:
    theta, phi = np.deg2rad(angles[:, 0]), np.deg2rad(angles[:, 1])
    return np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )


def angular_error_deg(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(prediction, axis=1)
    if not np.isfinite(prediction).all() or np.any(norm <= 0):
        raise ValueError("Nonfinite or zero predicted direction")
    unit = prediction / norm[:, None]
    return np.rad2deg(np.arccos(np.clip(np.sum(unit * truth, axis=1), -1, 1)))


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "q50": float(np.quantile(values, 0.5)),
        "q68": float(np.quantile(values, 0.68)),
    }


def summarize(
    mask: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    keys: np.ndarray,
    *,
    first_label: str,
    second_label: str,
    radius68: np.ndarray | None = None,
    radius95: np.ndarray | None = None,
) -> dict:
    count = int(mask.sum())
    if not count:
        return {"count": 0}
    a, b = first[mask], second[mask]
    result = {
        "count": count,
        "unique_primaries": int(len(np.unique(keys[mask]))),
        first_label: quantiles(a),
        second_label: quantiles(b),
        "fraction_first_better_eventwise": float(np.mean(a < b)),
    }
    if radius68 is not None and radius95 is not None:
        if not (np.isfinite(radius68[mask]).all() and np.isfinite(radius95[mask]).all()):
            raise ValueError("Nonfinite calibrated angle radii")
        result["angle_coverage68"] = float(np.mean(a <= radius68[mask]))
        result["angle_coverage95"] = float(np.mean(a <= radius95[mask]))
    return result


def bootstrap_unique_primary_delta(
    keys: np.ndarray, first: np.ndarray, second: np.ndarray, *, seed: int = 51
) -> dict:
    _, first_index = np.unique(keys, return_index=True)
    first_index.sort()
    a, b = first[first_index], second[first_index]
    rng = np.random.default_rng(seed)
    changes = np.empty((1000, 2))
    for i in range(len(changes)):
        sample = rng.integers(0, len(a), len(a))
        changes[i] = [
            np.quantile(a[sample], level) - np.quantile(b[sample], level) for level in (0.5, 0.68)
        ]
    return {
        "unique_primaries": int(len(first_index)),
        "first": quantiles(a),
        "second": quantiles(b),
        "delta_first_minus_second_ci95": {
            label: np.quantile(changes[:, j], [0.025, 0.975]).tolist()
            for j, label in enumerate(("q50", "q68"))
        },
    }


def read_quality(path: Path | None) -> tuple[np.ndarray, np.ndarray]:
    if path is None:
        return np.array([], dtype="S64"), np.array([], dtype="S64")
    with h5py.File(path) as h5:
        ids = np.concatenate(
            [np.asarray(h5[f"{split}/event_ids"], dtype="S64") for split in ("val", "test")]
        )
        passed = np.concatenate(
            [np.asarray(h5[f"{split}/quality_mask"], dtype=bool) for split in ("val", "test")]
        )
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate official-quality event IDs")
    return ids, ids[passed]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-data", type=Path, required=True)
    parser.add_argument("--reco-data", type=Path, required=True)
    parser.add_argument("--reference-gt", type=Path, required=True)
    parser.add_argument("--reference-pred", type=Path, required=True)
    parser.add_argument("--prediction-train", type=Path, required=True)
    parser.add_argument("--prediction-val", type=Path, required=True)
    parser.add_argument("--quality-sidecar", type=Path)
    parser.add_argument("--prediction-p090-train", type=Path)
    parser.add_argument("--prediction-p090-val", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if (args.prediction_p090_train is None) != (args.prediction_p090_val is None):
        raise ValueError("Specify both p090 prediction files or neither")

    with h5py.File(args.reference_gt) as gt, h5py.File(args.reference_pred) as pred:
        known = np.unique(
            np.concatenate(
                [
                    primary_keys(source[f"{split}/prime_prty/data"][:])
                    for source in (gt, pred)
                    for split in ("train", "val", "test")
                ]
            )
        )
    quality_ids, quality_pass_ids = read_quality(args.quality_sidecar)
    with h5py.File(args.reco_data) as reco:
        reco_ids = np.concatenate(
            [
                np.asarray(reco[f"{split}/ev_ids/data"], dtype="S64")
                for split in ("train", "val", "test")
            ]
        )
        reco_primaries = np.concatenate(
            [reco[f"{split}/prime_prty/data"][:] for split in ("train", "val", "test")]
        )
        reco_values = np.concatenate(
            [reco[f"{split}/reco_ev/data"][:] for split in ("train", "val", "test")]
        )
    if len(np.unique(reco_ids)) != len(reco_ids):
        raise ValueError("Duplicate standard-reco event IDs")
    order = np.argsort(reco_ids)
    sorted_ids = reco_ids[order]
    report = {
        "selection": "p>=0.70 predicted-signal hits; >=8 hits on >=2 strings",
        "primary_disjoint_from": "all train/val/test primaries in both internal 2020 GT and predicted-hit HDF5s",
        "official_quality_scope": (
            "Three available official reco cuts from the sidecar only, not ROOT 10/13 or full 13/13"
        ),
        "splits": {},
        "inputs": {key: str(value) for key, value in vars(args).items() if value is not None},
    }
    combined = []
    combined_p090 = []
    with h5py.File(args.pred_data) as pred_data:
        for split in ("train", "val"):
            ids = np.asarray(pred_data[f"{split}/ev_ids/data"], dtype="S64")
            primaries = pred_data[f"{split}/prime_prty/data"][:]
            primary = primary_keys(primaries)
            novel = ~np.isin(primary, known)
            frame = pd.read_csv(getattr(args, f"prediction_{split}"))
            if not np.array_equal(frame.event_id.to_numpy(dtype="S64"), ids):
                raise ValueError(f"Prediction/HDF5 event ID mismatch in {split}")
            positions = np.searchsorted(sorted_ids, ids)
            if np.any(positions == len(sorted_ids)) or not np.array_equal(
                sorted_ids[positions], ids
            ):
                raise ValueError(f"Missing standard-reco event ID in {split}")
            reco_index = order[positions]
            if not np.array_equal(reco_primaries[reco_index, :3], primaries[:, :3]):
                raise ValueError(f"Prediction/standard-reco truth mismatch in {split}")
            truth = vectors_from_angles(primaries[:, :2])
            nn_error = angular_error_deg(
                frame[["direction_x", "direction_y", "direction_z"]].to_numpy(), truth
            )
            reco_error = angular_error_deg(
                vectors_from_angles(reco_values[reco_index][:, [7, 8]]), truth
            )
            reco_hits = reco_values[reco_index, 4]
            selected_hits = frame.n_hits.to_numpy()
            r68 = frame.angle_r68_deg.to_numpy()
            r95 = frame.angle_r95_deg.to_numpy()
            result = {
                "source_events": int(len(ids)),
                "novel_primaries": summarize(
                    novel,
                    nn_error,
                    reco_error,
                    primary,
                    first_label="nn",
                    second_label="standard_reco",
                    radius68=r68,
                    radius95=r95,
                ),
                "novel_reco_nhits_ge20": summarize(
                    novel & (reco_hits >= 20),
                    nn_error,
                    reco_error,
                    primary,
                    first_label="nn",
                    second_label="standard_reco",
                    radius68=r68,
                    radius95=r95,
                ),
                "novel_selected_hits_8to11": summarize(
                    novel & (selected_hits < 12),
                    nn_error,
                    reco_error,
                    primary,
                    first_label="nn",
                    second_label="standard_reco",
                    radius68=r68,
                    radius95=r95,
                ),
            }
            if len(quality_ids):
                result["novel_quality_sidecar_available"] = summarize(
                    novel & np.isin(ids, quality_ids),
                    nn_error,
                    reco_error,
                    primary,
                    first_label="nn",
                    second_label="standard_reco",
                    radius68=r68,
                    radius95=r95,
                )
                result["novel_official_quality_pass"] = summarize(
                    novel & np.isin(ids, quality_pass_ids),
                    nn_error,
                    reco_error,
                    primary,
                    first_label="nn",
                    second_label="standard_reco",
                    radius68=r68,
                    radius95=r95,
                )
            if args.prediction_p090_train is not None:
                p090 = pd.read_csv(getattr(args, f"prediction_p090_{split}"))
                index = p090.source_event_index.to_numpy(dtype=np.int64)
                if (
                    np.any(index < 0)
                    or np.any(index >= len(ids))
                    or len(np.unique(index)) != len(index)
                    or not np.array_equal(p090.event_id.to_numpy(dtype="S64"), ids[index])
                ):
                    raise ValueError(f"p090/HDF5 source index mismatch in {split}")
                p090_error = angular_error_deg(
                    p090[["direction_x", "direction_y", "direction_z"]].to_numpy(),
                    truth[index],
                )
                common = novel[index]
                result["p090_novel_retention"] = float(common.sum() / novel.sum())
                result["p090_vs_p070_novel_common"] = summarize(
                    common,
                    p090_error,
                    nn_error[index],
                    primary[index],
                    first_label="p090",
                    second_label="p070",
                )
                result["p090_vs_p070_reused_common"] = summarize(
                    ~common,
                    p090_error,
                    nn_error[index],
                    primary[index],
                    first_label="p090",
                    second_label="p070",
                )
                combined_p090.append(
                    (primary[index][common], p090_error[common], nn_error[index][common])
                )
            combined.append(
                (primary[novel], nn_error[novel], reco_error[novel], r68[novel], r95[novel])
            )
            report["splits"][split] = result
    keys, nn, reco_error, r68, r95 = [
        np.concatenate([row[i] for row in combined]) for i in range(5)
    ]
    report["combined_novel"] = summarize(
        np.ones(len(nn), dtype=bool),
        nn,
        reco_error,
        keys,
        first_label="nn",
        second_label="standard_reco",
        radius68=r68,
        radius95=r95,
    )
    report["combined_novel_unique_primary_bootstrap"] = bootstrap_unique_primary_delta(
        keys, nn, reco_error
    )
    if combined_p090:
        keys90, p090_error, p070_error = [
            np.concatenate([row[i] for row in combined_p090]) for i in range(3)
        ]
        report["p090_vs_p070_combined_novel_common"] = summarize(
            np.ones(len(keys90), dtype=bool),
            p090_error,
            p070_error,
            keys90,
            first_label="p090",
            second_label="p070",
        )
        report["p090_vs_p070_unique_primary_bootstrap"] = bootstrap_unique_primary_delta(
            keys90, p090_error, p070_error, seed=117
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
