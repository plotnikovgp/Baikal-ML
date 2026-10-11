#!/usr/bin/env python3
"""Compare angular and line-point quality across hit thresholds on fixed IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THRESHOLDS = (("070", 0.70), ("075", 0.75), ("080", 0.80), ("090", 0.90))


def q68(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.68))


def paired_ci(baseline: np.ndarray, candidate: np.ndarray) -> list[float]:
    rng = np.random.default_rng(59)
    differences = np.empty(500)
    for i in range(len(differences)):
        sampled = rng.integers(len(baseline), size=len(baseline))
        differences[i] = q68(candidate[sampled]) - q68(baseline[sampled])
    return np.quantile(differences, [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.sweep_dir
    rows = []
    for particle in ("nuatm", "nue2"):
        results = {}
        for tag, _ in THRESHOLDS:
            with np.load(root / f"{particle}_quality_p{tag}.npz") as data:
                results[tag] = {
                    name: data[name].copy() for name in ("event_id", "angle_deg", "line_m")
                }
        common_ids = results["090"]["event_id"]
        baseline_angle = baseline_line = None
        for tag, threshold in THRESHOLDS:
            result = results[tag]
            positions = pd.Index(result["event_id"]).get_indexer(common_ids)
            if (positions < 0).any():
                raise ValueError(f"{particle} p{tag} misses p090 IDs")
            angle, line = result["angle_deg"][positions], result["line_m"][positions]
            if not np.isfinite(line).all():
                raise ValueError(f"{particle} contains invalid line points at p{tag}")
            if baseline_angle is None:
                baseline_angle, baseline_line = angle, line
            selected = json.loads((root / f"{particle}_quality_p{tag}.json").read_text())
            rows.append(
                {
                    "particle": particle,
                    "threshold": threshold,
                    "selected_unique_events": len(result["event_id"]),
                    "common_p090_events": len(common_ids),
                    "selected_angle_q50_deg": selected["all_unique"]["q50_deg"],
                    "selected_angle_q68_deg": selected["all_unique"]["q68_deg"],
                    "selected_line_q50_m": selected["line_m_valid"]["q50_m"],
                    "selected_line_q68_m": selected["line_m_valid"]["q68_m"],
                    "common_angle_q50_deg": float(np.median(angle)),
                    "common_angle_q68_deg": q68(angle),
                    "common_line_q50_m": float(np.median(line)),
                    "common_line_q68_m": q68(line),
                    "common_angle_q68_delta_ci95_deg": paired_ci(baseline_angle, angle),
                    "common_line_q68_delta_ci95_m": paired_ci(baseline_line, line),
                }
            )
    (root / "other_particles_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(root / "other_particles_summary.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for particle in ("nuatm", "nue2"):
        group = [row for row in rows if row["particle"] == particle]
        axes[0].plot(
            [row["threshold"] for row in group],
            [row["common_angle_q68_deg"] for row in group],
            "o-",
            label=particle,
        )
        axes[1].plot(
            [row["threshold"] for row in group],
            [row["common_line_q68_m"] for row in group],
            "o-",
            label=particle,
        )
    axes[0].set_ylabel("Angular q68 [degrees]")
    axes[1].set_ylabel("Transverse-line q68 [m]")
    for axis in axes:
        axis.set_xlabel("Signal probability threshold")
        axis.set_xticks([fraction for _, fraction in THRESHOLDS])
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(root / "other_particles_tradeoff.png", dpi=180)
    plt.close(figure)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
