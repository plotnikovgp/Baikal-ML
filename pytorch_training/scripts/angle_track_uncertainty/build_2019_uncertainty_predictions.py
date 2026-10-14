#!/usr/bin/env python3
"""Join out-of-season angle and truth-line errors for calibration plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from audit_2019_point_uncertainty import load_line_error
from calibrate_cross_season_angle import read_angles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--mc-data", type=Path, required=True)
    parser.add_argument("--truth-points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    ids, _, angle, sigma_angle = read_angles(args.predictions, args.mc_data)
    point_rows, line, sigma_line, excluded = load_line_error(args.predictions, args.truth_points)
    if excluded or not np.array_equal(ids, point_rows.event_id.to_numpy()):
        raise ValueError("Angle and line truth cohorts are not identical")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        event_id=ids,
        angle=angle.astype(np.float32),
        sigma_angle=sigma_angle.astype(np.float32),
        line=line.astype(np.float32),
        sigma_line=sigma_line.astype(np.float32),
        valid=np.ones(len(ids), dtype=bool),
    )
    print(f"Saved {len(ids):,} aligned 2019 events to {args.output}")


if __name__ == "__main__":
    main()
