#!/usr/bin/env python3
"""Compare physical hit summaries on identical p>=0.70 MC/experiment cuts.

The muatm-MC events are weighted only by their observed hit/string counts,
using the same rule as the angular domain audit.  This report is a covariate
diagnostic, not a claim about experimental direction accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_domain_predictions import matching_weights, weighted_quantile
from scipy.stats import wasserstein_distance

FEATURES = (
    ("charge_mean", "mean charge"),
    ("charge_q90", "90th-percentile charge"),
    ("time_mean_ns", "mean time [ns]"),
    ("time_span_ns", "time span [ns]"),
    ("time_std_ns", "time std [ns]"),
    ("x_mean_m", "mean x [m]"),
    ("y_mean_m", "mean y [m]"),
    ("z_mean_m", "mean z [m]"),
    ("xy_rms_m", "horizontal spread [m]"),
    ("z_std_m", "vertical spread [m]"),
    ("p_mean", "mean signal p"),
    ("p_below_080", "fraction p<0.80"),
)


def summarize(path: Path, split: str, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.event_id.duplicated().any():
        raise ValueError("Prediction CSV must have unique event IDs")
    if not frame.source_event_index.is_monotonic_increasing:
        frame = frame.sort_values("source_event_index", kind="stable")
    index = frame.source_event_index.to_numpy(dtype=np.int64)
    with h5py.File(path, "r") as h5:
        group = h5[split]
        starts = np.asarray(group["ev_starts/data"][: int(index[-1]) + 2], dtype=np.int64)
        count = int(starts[-1])
        normal = np.asarray(group["data/data"][:count], dtype=np.float32)
        mean = np.asarray(h5["norm_param/mean"], dtype=np.float32)
        std = np.asarray(h5["norm_param/std"], dtype=np.float32)
        raw = normal * std + mean
        probability = np.asarray(group["signal_probability/data"][:count], dtype=np.float32)
        ids = np.asarray(group["ev_ids/data"][: int(index[-1]) + 1])
    expected = np.char.decode(ids[index].astype("S"), "utf-8")
    if not np.array_equal(expected, frame.event_id.to_numpy(dtype=str)):
        raise ValueError("CSV IDs do not match the source HDF5 rows")
    rows = []
    for event_index in index:
        a, b = int(starts[event_index]), int(starts[event_index + 1])
        hit = raw[a:b]
        p = probability[a:b]
        if len(hit) < 8:
            raise ValueError(f"Unexpected event below 8-hit cut: {event_index}")
        xyz = hit[:, 2:5]
        center = xyz.mean(axis=0)
        delta = xyz - center
        xy_rms = np.sqrt(np.mean(np.sum(delta[:, :2] ** 2, axis=1)))
        rows.append(
            (
                float(hit[:, 0].mean()),
                float(np.quantile(hit[:, 0], 0.9)),
                float(hit[:, 1].mean()),
                float(np.ptp(hit[:, 1])),
                float(hit[:, 1].std()),
                float(center[0]),
                float(center[1]),
                float(center[2]),
                float(xy_rms),
                float(delta[:, 2].std()),
                float(p.mean()),
                float(np.mean(p < 0.80)),
            )
        )
    result = pd.DataFrame(rows, columns=[name for name, _ in FEATURES])
    result.insert(0, "event_id", frame.event_id.to_numpy())
    result.insert(1, "n_hits", frame.n_hits.to_numpy(dtype=np.int32))
    result.insert(2, "n_strings", frame.n_strings.to_numpy(dtype=np.int32))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-data", type=Path, required=True)
    parser.add_argument("--mc-data", type=Path, required=True)
    parser.add_argument("--exp-csv", type=Path, required=True)
    parser.add_argument("--mc-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    exp_frame = pd.read_csv(args.exp_csv)
    mc_frame = pd.read_csv(args.mc_csv).drop_duplicates("event_id", keep="first")
    exp = summarize(args.exp_data, "test", exp_frame)
    mc = summarize(args.mc_data, "val", mc_frame)
    if not exp.event_id.equals(exp_frame.event_id):
        exp = exp.set_index("event_id").loc[exp_frame.event_id].reset_index()
    if not mc.event_id.equals(mc_frame.event_id):
        mc = mc.set_index("event_id").loc[mc_frame.event_id].reset_index()
    weights, support = matching_weights(exp_frame, mc_frame)
    report = {
        "exp_events": len(exp),
        "muatm_mc_events": len(mc),
        "matching": support,
        "features": {},
    }
    for name, _label in FEATURES:
        e = exp[name].to_numpy()
        m = mc[name].to_numpy()
        exp_std = float(e.std())
        mc_std = float(
            np.sqrt(np.average((m - np.average(m, weights=weights)) ** 2, weights=weights))
        )
        w1 = float(wasserstein_distance(e, m, v_weights=weights))
        report["features"][name] = {
            "exp_q10_q50_q90": np.quantile(e, [0.1, 0.5, 0.9]).tolist(),
            "mc_weighted_q10_q50_q90": [weighted_quantile(m, weights, q) for q in (0.1, 0.5, 0.9)],
            "w1": w1,
            "w1_over_pooled_std": w1 / max((exp_std + mc_std) / 2, 1e-9),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exp.to_csv(args.output_dir / "exp_features.csv", index=False)
    mc.to_csv(args.output_dir / "muatm_mc_features.csv", index=False)
    (args.output_dir / "feature_shift.json").write_text(json.dumps(report, indent=2) + "\n")
    ranking = sorted(FEATURES, key=lambda pair: report["features"][pair[0]]["w1_over_pooled_std"])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        [label for _, label in ranking],
        [report["features"][name]["w1_over_pooled_std"] for name, _ in ranking],
    )
    ax.set_xlabel("Hit/string-matched W1 / pooled standard deviation")
    ax.set_title("Experimental vs muatm MC input-feature shift, p>=0.70")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "feature_shift.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
