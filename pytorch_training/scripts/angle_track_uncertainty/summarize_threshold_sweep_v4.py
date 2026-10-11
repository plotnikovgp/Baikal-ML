#!/usr/bin/env python3
"""Summarize threshold trade-offs, including paired MC quality on common IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THRESHOLDS = (("070", 0.70), ("075", 0.75), ("080", 0.80), ("085", 0.85), ("090", 0.90))


def q68(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.68)) if len(values) else float("nan")


def paired_interval(baseline: np.ndarray, candidate: np.ndarray, seed: int = 47) -> list[float]:
    if len(baseline) < 10:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    deltas = np.empty(500)
    for i in range(len(deltas)):
        selected = rng.integers(0, len(baseline), len(baseline))
        deltas[i] = q68(candidate[selected]) - q68(baseline[selected])
    return np.quantile(deltas, [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.sweep_dir
    mc = {}
    domain = {}
    exp = {}
    for tag, _ in THRESHOLDS:
        with np.load(root / f"mc_quality_p{tag}.npz") as data:
            ids = data["event_id"].astype(str)
            if len(np.unique(ids)) != len(ids):
                raise ValueError(f"Repeated MC event IDs at p{tag}")
            mc[tag] = {
                key: data[key].copy() for key in ("event_id", "angle_deg", "line_m", "single_track")
            }
        domain[tag] = json.loads((root / f"domain_p{tag}/domain_report.json").read_text())
        exp[tag] = pd.read_csv(root / f"exp_test_p{tag}.csv").set_index("event_id")
    common_domain = json.loads((root / "domain_common_ids_summary.json").read_text())
    if [row["threshold"] for row in common_domain] != [value for _, value in THRESHOLDS]:
        raise ValueError("Common-ID domain thresholds do not match sweep thresholds")

    # Since a higher hit threshold only removes events, this is the exact p>=.90
    # population.  Keep it fixed when comparing reconstruction quality.
    common = mc["090"]["event_id"]
    common_exp = exp["090"].index
    reference = None
    rows = []
    for domain_row, (tag, threshold) in zip(common_domain, THRESHOLDS):
        values = mc[tag]
        index = pd.Index(values["event_id"])
        locations = index.get_indexer(common)
        if (locations < 0).any():
            raise ValueError(f"p{tag} is missing p090 events")
        error = values["angle_deg"][locations]
        line_error = values["line_m"][locations]
        single = values["single_track"][locations]
        if reference is None:
            reference = error
            reference_single = single
            reference_line = line_error
        if not np.array_equal(single, reference_single):
            raise ValueError(f"Multiplicity labels changed at p{tag}")
        if not np.array_equal(np.isfinite(line_error), np.isfinite(reference_line)):
            raise ValueError(f"Valid track-point labels changed at p{tag}")
        angles = exp[tag].loc[common_exp][["direction_x", "direction_y", "direction_z"]]
        old_angles = exp["070"].loc[common_exp][["direction_x", "direction_y", "direction_z"]]
        dot = np.sum(angles.to_numpy() * old_angles.to_numpy(), axis=1)
        direction_change = np.rad2deg(np.arccos(np.clip(dot, -1, 1)))
        quality = json.loads((root / f"mc_quality_p{tag}.json").read_text())
        report = domain[tag]
        row = {
            "threshold": threshold,
            "exp_test_events": len(exp[tag]),
            "mc_unique_events": len(values["event_id"]),
            "common_mc_events": len(common),
            "common_mc_single_events": int(single.sum()),
            "mc_selected_q50_deg": quality["all_unique"]["q50_deg"],
            "mc_selected_q68_deg": quality["all_unique"]["q68_deg"],
            "mc_selected_valid_line_q68_m": quality["line_m_valid"]["q68_m"],
            "mc_common_q50_deg": float(np.median(error)),
            "mc_common_q68_deg": q68(error),
            "mc_common_single_q68_deg": q68(error[single]),
            "mc_common_multi_q68_deg": q68(error[~single]),
            "mc_common_valid_line_q68_m": q68(line_error[np.isfinite(line_error)]),
            "mc_common_valid_line_delta_ci95_m": paired_interval(
                reference_line[np.isfinite(reference_line)],
                line_error[np.isfinite(line_error)],
            ),
            "mc_common_q68_delta_vs_p070_deg": q68(error) - q68(reference),
            "mc_common_q68_delta_ci95_deg": paired_interval(reference, error),
            "mc_common_single_delta_ci95_deg": paired_interval(reference[single], error[single]),
            "mc_common_multi_delta_ci95_deg": paired_interval(reference[~single], error[~single]),
            "exp_common_direction_change_from_p070_q50_deg": float(np.median(direction_change)),
            "exp_common_direction_change_from_p070_q68_deg": q68(direction_change),
            "exp_mc_theta_w1_unweighted_deg": report["unweighted"]["theta_wasserstein_deg"],
            "exp_mc_theta_w1_hit_string_matched_deg": report["input_matched"][
                "theta_wasserstein_deg"
            ],
            "exp_mc_theta_w1_common_matched_deg": domain_row["theta_w1_hit_string_matched_deg"],
            "exp_mc_theta_w1_common_unweighted_deg": domain_row["theta_w1_unweighted_deg"],
            "exp_outside_mc_support_fraction": report["input_matching"][
                "exp_outside_mc_support_fraction"
            ],
        }
        rows.append(row)

    (root / "threshold_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(root / "threshold_summary.csv", index=False)
    x = [row["threshold"] for row in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(x, [row["mc_common_q68_deg"] for row in rows], "o-", label="all")
    ax[0].plot(x, [row["mc_common_single_q68_deg"] for row in rows], "o-", label="one truth track")
    ax[0].plot(x, [row["mc_common_multi_q68_deg"] for row in rows], "o-", label="multi/invalid")
    ax[0].set_ylabel("MC angular q68 [degrees]")
    ax[0].set_xlabel("Signal probability threshold")
    ax[0].set_title("Identical p≥0.90 MC events")
    ax[0].legend()
    ax[1].plot(
        x,
        [row["exp_mc_theta_w1_common_matched_deg"] for row in rows],
        "o-",
        label="hit/string matched",
    )
    ax[1].plot(
        x,
        [row["exp_mc_theta_w1_common_unweighted_deg"] for row in rows],
        "o--",
        label="unweighted",
    )
    ax[1].set_ylabel("Experiment–MC predicted θ W1 [degrees]")
    ax[1].set_xlabel("Signal probability threshold")
    ax[1].set_title("Identical p≥0.90 exp/MC events")
    ax[1].legend()
    for axis in ax:
        axis.grid(alpha=0.25)
        axis.set_xticks(x)
    fig.tight_layout()
    fig.savefig(root / "threshold_tradeoff.png", dpi=180)
    plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
