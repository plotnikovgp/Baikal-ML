#!/usr/bin/env python3
"""Bundle direction, track-point and uncertainty weights into one CPU-loadable file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import torch
from train_angle_signal import SetTransformerDirection
from train_track_anchor_v3 import DirectionTrackAnchor
from train_track_uncertainty_v2 import TrackUncertainty


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(config: dict) -> TrackUncertainty:
    backbone = SetTransformerDirection(
        config["hidden"],
        config["layers"],
        config["heads"],
        config["ff_mult"],
        config["dropout"],
        config["feature_mode"],
    )
    base = DirectionTrackAnchor(
        backbone, config["hidden"], config["dropout"], config["anchor_scale_m"]
    )
    return TrackUncertainty(base, config["hidden"], config["uncertainty_dropout"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angle-checkpoint", type=Path, required=True)
    parser.add_argument("--track-checkpoint", type=Path, required=True)
    parser.add_argument("--uncertainty-checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    angle = torch.load(args.angle_checkpoint, map_location="cpu", weights_only=False)
    track = torch.load(args.track_checkpoint, map_location="cpu", weights_only=False)
    uncertainty = torch.load(args.uncertainty_checkpoint, map_location="cpu", weights_only=False)
    metrics = json.loads(args.metrics.read_text())
    angle_args = angle["args"]
    track_args = track["args"]
    expected_angle = str(args.angle_checkpoint.resolve())
    if Path(track_args["init_checkpoint"]).resolve() != Path(expected_angle):
        raise ValueError("Track checkpoint was not initialized from the supplied angle checkpoint")
    expected_track = str(args.track_checkpoint.resolve())
    if Path(uncertainty["base_checkpoint"]).resolve() != Path(expected_track):
        raise ValueError("Uncertainty head was not trained on the supplied track checkpoint")

    config = {
        key: angle_args[key]
        for key in ("hidden", "layers", "heads", "ff_mult", "dropout", "feature_mode")
    }
    config["anchor_scale_m"] = float(track_args["anchor_scale_m"])
    config["uncertainty_dropout"] = 0.05
    model = build_model(config)
    model.base.load_state_dict(track["model"], strict=True)
    model.head.load_state_dict(uncertainty["head"], strict=True)
    model.eval()

    norm_group = "signal_norm_param" if angle_args.get("use_signal_norm", False) else "norm_param"
    with h5py.File(args.data, "r") as h5:
        if norm_group not in h5:
            raise KeyError(f"Missing {norm_group} in training data")
        mean = h5[f"{norm_group}/mean"][:].astype(float).tolist()
        std = h5[f"{norm_group}/std"][:].astype(float).tolist()
    if len(mean) != 5 or len(std) != 5 or any(value <= 0 for value in std):
        raise ValueError("Expected five hit features with positive normalization scales")
    factors = metrics["calibration_from_val"]
    for key in ("angle", "line"):
        if not {"68", "95"}.issubset(factors[key]):
            raise KeyError(f"Missing validation calibration for {key}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_config": config,
        "state_dict": model.state_dict(),
        "preprocessing": {
            "feature_order": ["charge", "time_ns", "x_m", "y_m", "z_m"],
            "input_norm_mean": mean,
            "input_norm_std": std,
            "max_seq_len": int(track_args["max_seq_len"]),
            "center_time": bool(angle_args.get("center_time", False)),
            "hit_selection": "signal-noise p>=0.70; >=8 selected hits on >=2 strings",
        },
        "calibration": {
            key: {level: float(factors[key][level]) for level in ("68", "95")}
            for key in ("angle", "line")
        },
        "scope": {
            "direction": "muatm, nuatm and nue2; all selected events",
            "track_point_and_uncertainty": (
                "trained/calibrated only where original MC has exactly one simulated track; "
                "most muatm events are multi-track and have no validated point target"
            ),
            "coordinates": "cluster-local hit coordinates in metres",
            "anchor_definition": (
                "point on truth line closest to the centroid of all retained hits"
            ),
        },
        "test_metrics": metrics.get("test", {}),
        "provenance": {
            "angle_checkpoint_sha256": sha256(args.angle_checkpoint),
            "track_checkpoint_sha256": sha256(args.track_checkpoint),
            "uncertainty_checkpoint_sha256": sha256(args.uncertainty_checkpoint),
            "uncertainty_step": int(uncertainty["step"]),
        },
    }
    torch.save(payload, args.output)
    reloaded = torch.load(args.output, map_location="cpu", weights_only=True)
    check = build_model(reloaded["model_config"])
    check.load_state_dict(reloaded["state_dict"], strict=True)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bytes": args.output.stat().st_size,
                "parameters": sum(p.numel() for p in check.parameters()),
                "format_version": reloaded["format_version"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
