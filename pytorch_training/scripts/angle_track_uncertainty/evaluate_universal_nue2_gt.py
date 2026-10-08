#!/usr/bin/env python3
"""Evaluate the universal bundle on the exact nue2 GT-signal test and anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from bundle_universal_model import build_model
from infer_unified_model import read_batch
from train_angle_signal import direction_from_angles
from train_track_uncertainty_v2 import residuals


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        f"q{int(100 * probability)}": float(np.quantile(values, probability))
        for probability in (0.5, 0.68, 0.9, 0.95)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no visible GPU is available")
    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(bundle["model_config"])
    model.load_state_dict(bundle["state_dict"], strict=True)
    model = model.to(device).eval()
    fields = {key: [] for key in ("angle", "line", "sigma_angle", "sigma_line", "valid")}
    with h5py.File(args.data, "r") as data, h5py.File(args.anchors, "r") as anchors:
        total = len(data["test/ev_ids/data"])
        if len(anchors["test/event_ids"]) != total:
            raise ValueError("GT-hit test and anchor files have different event counts")
        if args.max_events is not None:
            total = min(total, args.max_events)
        for begin in range(0, total, args.batch_size):
            end = min(begin + args.batch_size, total)
            ids, _lengths, _strings, x, mask, _centers = read_batch(
                data, "test", begin, end, bundle["preprocessing"]
            )
            if not np.array_equal(ids, anchors["test/event_ids"][begin:end]):
                raise ValueError(f"Event-ID mismatch at rows {begin}:{end}")
            truth_angles = np.asarray(data["test/prime_prty/data"][begin:end, :2])
            truth_direction = direction_from_angles(torch.from_numpy(truth_angles)).to(device)
            truth_offset = torch.from_numpy(
                np.asarray(anchors["test/anchor_offset_m"][begin:end], dtype=np.float32)
            ).to(device)
            valid = np.asarray(anchors["test/valid"][begin:end], dtype=bool)
            with (
                torch.inference_mode(),
                torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"),
            ):
                direction, offset, log_angle, log_line = model(
                    torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device)
                )
            angle, line, _ = residuals(
                direction.float(), offset.float(), truth_direction, truth_offset
            )
            for name, values in {
                "angle": angle,
                "line": line,
                "sigma_angle": log_angle.exp(),
                "sigma_line": log_line.exp(),
            }.items():
                fields[name].append(values.cpu().numpy())
            fields["valid"].append(valid)
            if end % 100_000 < args.batch_size or end == total:
                print(f"evaluated {end:,}/{total:,}", flush=True)
    pred = {key: np.concatenate(chunks) for key, chunks in fields.items()}
    if not np.all(pred["valid"]):
        raise ValueError("The reference nue2 test unexpectedly contains invalid anchors")
    factors = bundle["calibration"]
    result = {
        "test_events": total,
        "input": "nue2 GT-signal hits with >=8 hits on >=2 strings",
        "direction_error_deg": quantiles(pred["angle"]),
        "transverse_anchor_error_m": quantiles(pred["line"]),
        "calibrated_coverage_on_gt_input": {
            key: {
                level: float(np.mean(pred[error] <= pred[sigma] * factors[key][level]))
                for level in ("68", "95")
            }
            for key, error, sigma in (
                ("angle", "angle", "sigma_angle"),
                ("line", "line", "sigma_line"),
            )
        },
        "note": (
            "This is out-of-training-input evaluation of the universal model on "
            "the exact GT-hit nue2 test. Its calibration was fitted on "
            "all-particle p>=0.70 predicted-signal hits, not on this GT input."
        ),
    }
    if args.reference_predictions:
        with np.load(args.reference_predictions) as reference:
            if len(reference["angle"]) != total or len(reference["line"]) != total:
                raise ValueError("Reference predictions do not match the test length")
            result["nue2_specialist_reference"] = {
                "direction_error_deg": quantiles(reference["angle"]),
                "transverse_anchor_error_m": quantiles(reference["line"]),
                "universal_better_fraction_angle": float(
                    np.mean(pred["angle"] < reference["angle"])
                ),
                "universal_better_fraction_anchor": float(
                    np.mean(pred["line"] < reference["line"])
                ),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.output.with_suffix(".npz"), **pred)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
