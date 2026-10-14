#!/usr/bin/env python3
"""Test frozen angle-sigma quality cuts across seasons and on data/MC theta.

Cut values are chosen from 2019-A without using ground-truth error. Direction
quality is evaluated on disjoint 2019-B. Experimental data have no truth;
their acceptance and matched-muatm theta distribution are diagnostics only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from audit_primary_overlap import primary_keys
from audit_theta_bias_crossseason import vectors
from match_domain_covariates import measure


def load_p070(csv_path: Path, h5_path: Path) -> dict[str, np.ndarray]:
    frame = pd.read_csv(csv_path)
    if frame.event_id.duplicated().any():
        raise ValueError(f"Duplicate prediction IDs in {csv_path}")
    with h5py.File(h5_path) as h5:
        ids = h5["test/ev_ids/data"][:]
        truth = h5["test/prime_prty/data"][:]
    lookup = {value.decode(): i for i, value in enumerate(ids)}
    index = np.array([lookup.get(value, -1) for value in frame.event_id], dtype=np.int64)
    if np.any(index < 0):
        raise ValueError(f"Prediction/HDF5 IDs mismatch in {csv_path}")
    return {
        "event_id": frame.event_id.to_numpy(dtype=str),
        "truth": truth[index, :2].astype(np.float64),
        "primary": truth[index, :3].copy(),
        "theta": frame.theta_deg.to_numpy(dtype=np.float64),
        "phi": frame.phi_deg.to_numpy(dtype=np.float64),
        "hits": frame.n_hits.to_numpy(dtype=np.int64),
        "sigma": frame.sigma_angle_deg.to_numpy(dtype=np.float64),
    }


def errors(rows: dict[str, np.ndarray]) -> np.ndarray:
    a = vectors(rows["theta"], rows["phi"])
    b = vectors(rows["truth"][:, 0], rows["truth"][:, 1])
    return np.rad2deg(np.arccos(np.clip((a * b).sum(axis=1), -1.0, 1.0)))


def metrics(error: np.ndarray, mask: np.ndarray) -> dict:
    if not mask.any():
        return {"n": 0}
    return {
        "n": int(mask.sum()),
        "fraction": float(mask.mean()),
        "q50_deg": float(np.quantile(error[mask], 0.5)),
        "q68_deg": float(np.quantile(error[mask], 0.68)),
    }


def bootstrap_selection_difference(
    error: np.ndarray, first: np.ndarray, second: np.ndarray, *, seed: int = 331
) -> dict:
    rng = np.random.default_rng(seed)
    changes = np.empty((300, 2))
    for i in range(len(changes)):
        sample = rng.integers(0, len(error), len(error))
        a, b = error[sample][first[sample]], error[sample][second[sample]]
        changes[i] = [np.quantile(a, level) - np.quantile(b, level) for level in (0.5, 0.68)]
    return {
        "delta_sigma_minus_hit_ci95_q50_deg": np.quantile(changes[:, 0], [0.025, 0.975]).tolist(),
        "delta_sigma_minus_hit_ci95_q68_deg": np.quantile(changes[:, 1], [0.025, 0.975]).tolist(),
    }


def joined(features: Path, predictions: Path, *, deduplicate: bool) -> pd.DataFrame:
    feat = pd.read_csv(features).set_index("event_id", verify_integrity=True)
    pred = pd.read_csv(predictions)
    if deduplicate:
        pred = pred.drop_duplicates("event_id", keep="first")
    pred = pred.set_index("event_id", verify_integrity=True)
    if not feat.index.isin(pred.index).all():
        raise ValueError(f"Missing prediction for {features}")
    aligned = pred.loc[feat.index]
    if not (
        np.array_equal(feat.n_hits.to_numpy(), aligned.n_hits.to_numpy())
        and np.array_equal(feat.n_strings.to_numpy(), aligned.n_strings.to_numpy())
    ):
        raise ValueError(f"Feature/prediction hit topology mismatch for {features}")
    result = feat.join(pred[["theta_deg", "sigma_angle_deg"]], how="left", validate="one_to_one")
    if result[["theta_deg", "sigma_angle_deg"]].isna().any().any():
        raise ValueError(f"Nonfinite prediction for {features}")
    return result


def novel_2020(
    data_path: Path,
    reference_gt: Path,
    reference_pred: Path,
    train_predictions: Path,
    val_predictions: Path,
) -> dict[str, np.ndarray]:
    with h5py.File(reference_gt) as gt, h5py.File(reference_pred) as pred:
        known = np.unique(
            np.concatenate(
                [
                    primary_keys(source[f"{split}/prime_prty/data"][:])
                    for source in (gt, pred)
                    for split in ("train", "val", "test")
                ]
            )
        )
    errors_by_split, scores_by_split = [], []
    with h5py.File(data_path) as h5:
        for split, csv_path in (("train", train_predictions), ("val", val_predictions)):
            ids = np.asarray(h5[f"{split}/ev_ids/data"], dtype="S64")
            primaries = h5[f"{split}/prime_prty/data"][:]
            mask = ~np.isin(primary_keys(primaries), known)
            frame = pd.read_csv(csv_path)
            if not np.array_equal(frame.event_id.to_numpy(dtype="S64"), ids):
                raise ValueError(f"2020 novel prediction/HDF5 IDs differ in {split}")
            truth = primaries[mask, :2].astype(np.float64)
            actual = vectors(truth[:, 0], truth[:, 1])
            predicted = vectors(frame.theta_deg.to_numpy()[mask], frame.phi_deg.to_numpy()[mask])
            errors_by_split.append(
                np.rad2deg(np.arccos(np.clip((actual * predicted).sum(axis=1), -1, 1)))
            )
            scores_by_split.append(frame.sigma_angle_deg.to_numpy()[mask])
    return {"error": np.concatenate(errors_by_split), "sigma": np.concatenate(scores_by_split)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "a_csv",
        "a_h5",
        "b_csv",
        "b_h5",
        "exp_features",
        "mc_features",
        "exp_predictions",
        "mc_predictions",
        "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    for name in (
        "novel_data",
        "reference_gt",
        "reference_pred",
        "novel_train_predictions",
        "novel_val_predictions",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    a = load_p070(args.a_csv, args.a_h5)
    b = load_p070(args.b_csv, args.b_h5)
    a_score, b_score = a["sigma"], b["sigma"]
    if not (np.isfinite(a_score).all() and np.isfinite(b_score).all()):
        raise ValueError("Nonfinite angle sigma")
    a_keys = np.ascontiguousarray(a["primary"]).view("V12").ravel()
    b_keys = np.ascontiguousarray(b["primary"]).view("V12").ravel()
    if np.intersect1d(a_keys, b_keys).size:
        raise ValueError("2019-A and 2019-B share generated primaries")
    exp = joined(args.exp_features, args.exp_predictions, deduplicate=False)
    mc = joined(args.mc_features, args.mc_predictions, deduplicate=True)
    novel_args = (
        args.novel_data,
        args.reference_gt,
        args.reference_pred,
        args.novel_train_predictions,
        args.novel_val_predictions,
    )
    if any(value is not None for value in novel_args) and not all(
        value is not None for value in novel_args
    ):
        raise ValueError("Specify all novel-2020 inputs or none")
    novel = novel_2020(*novel_args) if all(value is not None for value in novel_args) else None
    _, mc_weights = measure(exp, mc, ("z_std_m",), return_weights=True)
    b_error = errors(b)
    report = {
        "method": "Unchanged frozen v3 p>=0.70 direction and raw angle sigma; A-set score quantiles only",
        "inputs": {
            name: str(value)
            for name, value in vars(args).items()
            if name != "output" and value is not None
        },
        "b_all": metrics(b_error, np.ones(len(b_error), dtype=bool)),
        "exp_events": int(len(exp)),
        "muatm_mc_events": int(len(mc)),
        "full_exp_muatm_theta_match": measure(exp, mc, ("z_std_m",)),
        "cutoffs": {},
    }
    for retained_a in (0.5, 0.75):
        cutoff = float(np.quantile(a_score, retained_a))
        a_keep = a_score <= cutoff
        b_keep = b_score <= cutoff
        exp_keep = exp.sigma_angle_deg.to_numpy() <= cutoff
        mc_keep = mc.sigma_angle_deg.to_numpy() <= cutoff
        exp_selected = exp.iloc[np.flatnonzero(exp_keep)]
        mc_selected = mc.iloc[np.flatnonzero(mc_keep)]
        domain = measure(exp_selected, mc_selected, ("z_std_m",))
        # A-only multiplicity comparator, with no B label or threshold tuning.
        hit_cut = int(np.quantile(a["hits"], 1.0 - retained_a))
        hit_keep = b["hits"] >= hit_cut
        equal_hit_keep = np.zeros(len(b_error), dtype=bool)
        hit_order = np.lexsort((b["event_id"], -b["hits"]))
        equal_hit_keep[hit_order[: int(b_keep.sum())]] = True
        report["cutoffs"][str(retained_a)] = {
            "sigma_cut_deg": cutoff,
            "a_acceptance": float(a_keep.mean()),
            "b_retained": metrics(b_error, b_keep),
            "b_rejected": metrics(b_error, ~b_keep),
            "a_hit_count_cut": hit_cut,
            "b_hit_count_comparator": metrics(b_error, hit_keep),
            "b_equal_retention_hit_count_comparator": metrics(b_error, equal_hit_keep),
            "b_sigma_vs_equal_retention_hit_bootstrap": bootstrap_selection_difference(
                b_error, b_keep, equal_hit_keep
            ),
            "b_within_hit_bins": {
                label: {
                    "bin_n": int(in_bin.sum()),
                    "retained_fraction_within_bin": float((b_keep & in_bin).sum() / in_bin.sum()),
                    "retained": metrics(b_error, b_keep & in_bin),
                    "rejected": metrics(b_error, ~b_keep & in_bin),
                }
                for label, in_bin in (
                    ("8-11", b["hits"] < 12),
                    ("12-19", (b["hits"] >= 12) & (b["hits"] < 20)),
                    ("20+", b["hits"] >= 20),
                )
            },
            "exp_acceptance": float(exp_keep.mean()),
            "muatm_mc_acceptance_unweighted": float(mc_keep.mean()),
            "muatm_mc_acceptance_matched": float(mc_weights[mc_keep].sum()),
            "selected_exp_muatm_theta_match": domain,
        }
        if novel is not None:
            novel_keep = novel["sigma"] <= cutoff
            report["cutoffs"][str(retained_a)]["angle_model_primary_disjoint_2020_reco"] = {
                "cohort": "Angle-model-primary-disjoint, but signal/noise training overlap unaudited",
                "all": metrics(novel["error"], np.ones(len(novel["error"]), dtype=bool)),
                "retained": metrics(novel["error"], novel_keep),
                "rejected": metrics(novel["error"], ~novel_keep),
            }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
