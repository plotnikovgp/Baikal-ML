#!/usr/bin/env python3
"""Summarize leakage-free v4 large-error events by observed geometry and MC labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PARTICLES = ("muatm", "nuatm", "nue2")
FEATURES = (
    "n_hits",
    "n_strings",
    "hit_span_m",
    "z_span_m",
    "p_mean",
    "label_negative_fraction",
    "label_positive_fraction",
    "label_zero_fraction",
    "tres_abs_median_ns",
    "tres_abs_gt10_fraction",
    "photon_mad_pred_ns",
    "photon_mad_truth_ns",
    "true_theta_deg",
)
SPAN_BINS = (
    ("<100", 0, 100),
    ("100–150", 100, 150),
    ("150–200", 150, 200),
    ("200–300", 200, 300),
    ("300+", 300, np.inf),
)
NEGATIVE_BINS = (
    ("<0.25", 0, 0.25),
    ("0.25–0.50", 0.25, 0.5),
    ("0.50–0.75", 0.5, 0.75),
    ("0.75+", 0.75, 1.01),
)


def risk(frame: pd.DataFrame, selection: pd.Series, threshold: float) -> dict[str, float]:
    cohort = frame.loc[selection]
    severe = int((cohort.angular_error_deg >= threshold).sum())
    return {
        "events": int(len(cohort)),
        "errors": severe,
        "fraction": float(severe / len(cohort)) if len(cohort) else float("nan"),
    }


def median(value: pd.Series) -> float:
    return float(value.median()) if len(value) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.events).drop_duplicates("event_id", keep="first")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "unique_events": len(frame),
        "error_threshold_deg": 20,
        "interpretation": "Associations on MC test, not causal effects or experimental resolution",
        "particle": {},
        "all": {},
    }
    for name in ("all",) + PARTICLES:
        part = frame if name == "all" else frame.loc[frame.particle == name]
        severe = part.loc[part.angular_error_deg >= 20]
        ordinary = part.loc[part.angular_error_deg < 20]
        result = {
            "events": len(part),
            "error_ge_20": risk(part, part.angular_error_deg >= 0, 20),
            "error_ge_45": risk(part, part.angular_error_deg >= 0, 45),
            "error_ge_90": risk(part, part.angular_error_deg >= 0, 90),
            "median_features_error_lt20": {key: median(ordinary[key]) for key in FEATURES},
            "median_features_error_ge20": {key: median(severe[key]) for key in FEATURES},
            "span_bins": {},
            "negative_label_bins": {},
            "noise_presence": {},
        }
        for label, lower, upper in SPAN_BINS:
            result["span_bins"][label] = risk(
                part, (part.hit_span_m >= lower) & (part.hit_span_m < upper), 20
            )
        for label, lower, upper in NEGATIVE_BINS:
            result["negative_label_bins"][label] = risk(
                part,
                (part.label_negative_fraction >= lower) & (part.label_negative_fraction < upper),
                20,
            )
        for label, selection in (
            ("no_label_zero", part.label_zero_fraction == 0),
            ("any_label_zero", part.label_zero_fraction > 0),
            (
                "short_lt120_no_label_zero",
                (part.hit_span_m < 120) & (part.label_zero_fraction == 0),
            ),
            (
                "short_lt120_any_label_zero",
                (part.hit_span_m < 120) & (part.label_zero_fraction > 0),
            ),
        ):
            result["noise_presence"][label] = risk(part, selection, 20)
        result["severe_theta_abs_lt10_fraction"] = float(np.mean(severe.theta_error_abs_deg < 10))
        if name == "all":
            report["all"] = result
        else:
            report["particle"][name] = result

    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    colors = {"muatm": "#cc4455", "nuatm": "#347ab7", "nue2": "#56a36b"}
    for name in PARTICLES:
        row = report["particle"][name]
        for ax, field, bins in (
            (axes[0], "span_bins", SPAN_BINS),
            (axes[1], "negative_label_bins", NEGATIVE_BINS),
        ):
            y = [row[field][label]["fraction"] * 100 for label, *_ in bins]
            ax.plot(range(len(y)), y, "o-", lw=2, ms=5, label=name, color=colors[name])
    axes[0].set_xticks(range(len(SPAN_BINS)), [label for label, *_ in SPAN_BINS])
    axes[0].set_xlabel("Maximum hit separation [m]")
    axes[1].set_xticks(range(len(NEGATIVE_BINS)), [label for label, *_ in NEGATIVE_BINS])
    axes[1].set_xlabel("Fraction of hits with negative MC label")
    for ax in axes:
        ax.set_ylabel("Fraction with angular error ≥20° [%]")
        ax.grid(alpha=0.25)
        ax.set_ylim(bottom=0)
    axes[0].legend(title="Particle", frameon=False)
    fig.suptitle("Where v4 makes large angular errors · independent MC test")
    fig.savefig(args.output_dir / "risk_by_span_and_hit_label.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"events": len(frame), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
