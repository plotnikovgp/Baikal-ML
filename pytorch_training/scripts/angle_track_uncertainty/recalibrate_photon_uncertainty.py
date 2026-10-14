#!/usr/bin/env python3
"""Recalibrate frozen v4 uncertainty outputs after the photon-line correction.

No model parameter is fitted here.  The existing uncertainty head and photon
direction/point correction are frozen; only multiplicative 68%/95% radii are
estimated from the validation IDs reserved for calibration.  An independent
ID-masked test split measures coverage and uncertainty ranking before a
single-file CPU bundle is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from bundle_universal_model import sha256
from photon_uncertainty_bundle import build_model, load_frozen_sources
from train_angle_signal import make_loader
from train_mixed_uncertainty import predict_nonempty
from train_mixed_universal import AdaptedTrackDataset
from train_track_uncertainty_v2 import calibrate, report


def loader(
    data: Path,
    anchors: Path,
    split: str,
    keep: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    batch_size: int,
    workers: int,
    max_seq_len: int,
):
    dataset = AdaptedTrackDataset(
        str(data),
        split,
        batch_size,
        max_seq_len,
        False,
        None,
        False,
        anchor_path=str(anchors),
        keep=keep,
        expected_mean=mean,
        expected_std=std,
        single_muatm_only=True,
    )
    return make_loader(dataset, workers, False, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--photon-checkpoint", type=Path, required=True)
    parser.add_argument("--pred-data", type=Path, required=True)
    parser.add_argument("--pred-anchors", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("Invalid batch size or worker count")
    torch.set_num_threads(4)
    model, original, config = load_frozen_sources(args.bundle, args.photon_checkpoint)
    model = model.to(args.device).eval()
    prep = original["preprocessing"]
    mean = np.asarray(prep["input_norm_mean"], dtype=np.float32)
    std = np.asarray(prep["input_norm_std"], dtype=np.float32)
    with np.load(args.masks) as masks_file:
        masks = {key: masks_file[key].copy() for key in masks_file.files}
    validation_index = np.arange(len(masks["pred_val"]))
    calibration_mask = masks["pred_val"] & (validation_index % 2 == 1)
    if not calibration_mask.any() or not masks["pred_test"].any():
        raise ValueError("Empty calibration or independent test selection")
    common_loader_args = {
        "batch_size": args.batch_size,
        "workers": args.workers,
        "max_seq_len": int(prep["max_seq_len"]),
    }
    calibration = predict_nonempty(
        model,
        loader(
            args.pred_data,
            args.pred_anchors,
            "val",
            calibration_mask,
            mean,
            std,
            **common_loader_args,
        ),
        torch.device(args.device),
    )
    factors = calibrate(calibration)
    test = predict_nonempty(
        model,
        loader(
            args.pred_data,
            args.pred_anchors,
            "test",
            masks["pred_test"],
            mean,
            std,
            **common_loader_args,
        ),
        torch.device(args.device),
    )
    metrics = {
        "model": "frozen v4 uncertainty head + frozen photon direction/point correction",
        "calibration_from_val": factors,
        "original_v4_calibration": original["calibration"],
        "calibration_count": int(len(calibration["angle"])),
        "calibration_valid_track_count": int(calibration["valid"].sum()),
        "test_count": int(len(test["angle"])),
        "test_valid_track_count": int(test["valid"].sum()),
        "test": {"predicted_hits_single_muatm": report(test, factors)},
        "test_with_stale_v4_factors": {
            "angle_coverage68": float(
                np.mean(
                    test["angle"] <= original["calibration"]["angle"]["68"] * test["sigma_angle"]
                )
            ),
            "angle_coverage95": float(
                np.mean(
                    test["angle"] <= original["calibration"]["angle"]["95"] * test["sigma_angle"]
                )
            ),
            "line_coverage68": float(
                np.mean(
                    test["line"][test["valid"]]
                    <= original["calibration"]["line"]["68"] * test["sigma_line"][test["valid"]]
                )
            ),
            "line_coverage95": float(
                np.mean(
                    test["line"][test["valid"]]
                    <= original["calibration"]["line"]["95"] * test["sigma_line"][test["valid"]]
                )
            ),
        },
        "selection": (
            "Predicted-hit p>=0.70; >=8 hits on >=2 strings; single-track muatm only. "
            "Calibration uses odd-index, ID-masked validation rows disjoint from the "
            "original uncertainty-head selection half; test uses independent ID masks."
        ),
        "warning": (
            "Experimental truth and multi-track muatm uncertainty coverage are unknown. "
            "The independent test HDF5 includes repeated IDs; row counts are not unique IDs."
        ),
    }
    args.output_dir.mkdir(parents=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.output_dir / "test_predictions.npz", **test)
    scope = dict(original["scope"])
    scope["uncertainty_calibration"] = (
        "68%/95% radii recalibrated for frozen photon-corrected direction/line "
        "on disjoint predicted-hit p>=0.70 MC validation; experimental and "
        "multi-track-muatm coverage unvalidated"
    )
    payload = {
        "format_version": 2,
        "model_type": "photon_corrected_uncertainty",
        "model_config": config,
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "preprocessing": prep,
        "calibration": {
            key: {level: float(value[level]) for level in ("68", "95")}
            for key, value in factors.items()
        },
        "scope": scope,
        "test_metrics": metrics["test"],
        "provenance": {
            "base_bundle_sha256": sha256(args.bundle),
            "photon_checkpoint_sha256": sha256(args.photon_checkpoint),
            "calibration_source": str(args.pred_data),
            "calibration_mask": str(args.masks),
            "calibration_rows": int(len(calibration["angle"])),
            "no_fitted_parameters": True,
        },
    }
    bundle_path = args.output_dir / "universal_angle_track_uncertainty_photon_v1.pt"
    torch.save(payload, bundle_path)
    reloaded = torch.load(bundle_path, map_location="cpu", weights_only=True)
    rebuilt = build_model(reloaded["model_config"])
    rebuilt.load_state_dict(reloaded["state_dict"], strict=True)
    print(
        json.dumps(
            {
                "bundle": str(bundle_path),
                "bytes": bundle_path.stat().st_size,
                "calibration_count": metrics["calibration_count"],
                "test_count": metrics["test_count"],
                "factors": factors,
                "test_all": metrics["test"]["predicted_hits_single_muatm"]["all"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
