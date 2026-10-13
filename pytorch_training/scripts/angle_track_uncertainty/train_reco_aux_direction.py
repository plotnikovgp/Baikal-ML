#!/usr/bin/env python3
"""Conservative direction-only adaptation on disjoint external reco-MC files.

Only the final two direction rows of the photon residual head can change.
The point rows and the shared feature extractor remain bitwise unchanged.
Independent internal-MC validation is a checkpoint gate, not a train source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from bundle_universal_model import build_model
from photon_residual_upgrade import PhotonResidualUpgrade
from torch import nn
from train_angle_signal import (
    BatchedAngleDataset,
    angular_errors_degrees,
    infinite_batches,
    loss_function,
    make_loader,
    seed_everything,
)


def primary_keys(values: np.ndarray) -> np.ndarray:
    """Exact float32 (theta, phi, energy) fingerprint of generated primaries."""
    return np.ascontiguousarray(values[:, :3]).view("V12").reshape(-1)


class AngleData(BatchedAngleDataset):
    def __init__(self, path, split, events_per_batch, mean, std, *, keep=None):
        super().__init__(path, split, events_per_batch, max_seq_len=128)
        self.target_mean = np.asarray(mean, dtype=np.float32)
        self.target_std = np.asarray(std, dtype=np.float32)
        self.renormalize = not (
            np.allclose(self.stored_mean, self.target_mean)
            and np.allclose(self.stored_std, self.target_std)
        )
        if keep is not None and len(keep) < self.n_events:
            raise ValueError(f"Selection mask too short: {split}")
        self.keep = keep

    def __getitem__(self, batch_index):
        result = super().__getitem__(batch_index)
        if self.keep is None:
            return result
        start = batch_index * self.events_per_batch
        selection = torch.from_numpy(
            np.asarray(self.keep[start : start + len(result[0])], dtype=bool).copy()
        )
        return tuple(item[selection] for item in result)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    values = []
    for x, target, mask, _particle in loader:
        if not len(x):
            continue
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            direction, _point = model(x.to(device), mask.to(device))
        values.append(angular_errors_degrees(direction.float(), target.to(device)).cpu().numpy())
    errors = np.concatenate(values)
    return {
        "count": int(len(errors)),
        "q50": float(np.quantile(errors, 0.50)),
        "q68": float(np.quantile(errors, 0.68)),
        "q95": float(np.quantile(errors, 0.95)),
    }


def loader(path, split, batch_size, mean, std, workers, seed, *, keep=None, shuffle=False):
    data = AngleData(path, split, batch_size, mean, std, keep=keep)
    return make_loader(data, workers, shuffle, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--reco-data", type=Path, required=True)
    parser.add_argument("--internal-pred-data", type=Path, required=True)
    parser.add_argument("--internal-pred-anchors", type=Path, required=True)
    parser.add_argument("--gt-data", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-clean-reco-val", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=914)
    args = parser.parse_args()
    if min(args.steps, args.val_every, args.batch_size, args.learning_rate) <= 0:
        raise ValueError("Training counts and learning rate must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n"
    )
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
    full = build_model(bundle["model_config"])
    full.load_state_dict(bundle["state_dict"], strict=True)
    mean = np.asarray(bundle["preprocessing"]["input_norm_mean"], dtype=np.float32)
    std = np.asarray(bundle["preprocessing"]["input_norm_std"], dtype=np.float32)
    model = PhotonResidualUpgrade(
        full.base, torch.from_numpy(mean), torch.from_numpy(std), hidden=64
    ).to(device)
    saved = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"], strict=True)
    model.requires_grad_(False)
    final = model.hit_head[-1]
    assert isinstance(final, nn.Linear) and final.out_features == 4
    final.requires_grad_(True)
    point_weight_initial = final.weight.detach()[2:].clone()
    point_bias_initial = final.bias.detach()[2:].clone()
    optimizer = torch.optim.AdamW(final.parameters(), lr=args.learning_rate, weight_decay=0.0)

    with np.load(args.masks) as masks:
        pred_keep = np.asarray(masks["pred_val"], dtype=bool).copy()
        gt_keep = np.asarray(masks["gt_val"], dtype=bool).copy()
    with (
        h5py.File(args.internal_pred_data) as data,
        h5py.File(args.internal_pred_anchors) as anchors,
        h5py.File(args.gt_data) as gt,
        h5py.File(args.reco_data) as reco,
    ):
        event_ids = data["val/ev_ids/data"][:]
        anchor_ids = anchors["val/event_ids"][:]
        if not np.array_equal(event_ids, anchor_ids):
            raise ValueError("Internal predicted-hit anchor IDs do not match")
        single_particle_track = np.asarray(anchors["val/valid"][:], dtype=bool)
        is_muatm = np.fromiter(
            (event_id.startswith(b"muatm_") for event_id in event_ids),
            dtype=bool,
            count=len(event_ids),
        )
        pred_keep &= ~is_muatm | single_particle_track
        internal_train_primaries = np.union1d(
            primary_keys(data["train/prime_prty/data"][:]),
            primary_keys(gt["train/prime_prty/data"][:]),
        )
        all_train_primaries = np.union1d(
            internal_train_primaries,
            primary_keys(reco["train/prime_prty/data"][:]),
        )
        pred_keep &= ~np.isin(primary_keys(data["val/prime_prty/data"][:]), all_train_primaries)
        gt_keep &= ~np.isin(primary_keys(gt["val/prime_prty/data"][:]), all_train_primaries)
        reco_clean_keep = ~np.isin(
            primary_keys(reco["val/prime_prty/data"][:]), all_train_primaries
        )
    gt_keep &= np.arange(len(gt_keep)) % 6 == 0
    print(
        "primary_disjoint_validation "
        + json.dumps(
            {
                "reco_val": int(reco_clean_keep.sum()),
                "internal_pred_val": int(pred_keep.sum()),
                "gt_clean_val": int(gt_keep.sum()),
                "min_clean_reco_val": args.min_clean_reco_val,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    reco_train = infinite_batches(
        loader(
            args.reco_data,
            "train",
            args.batch_size,
            mean,
            std,
            args.workers,
            args.seed + 1,
            shuffle=True,
        )
    )
    reco_val = loader(
        args.reco_data, "val", args.batch_size, mean, std, args.workers, args.seed + 2
    )
    reco_clean_val = loader(
        args.reco_data,
        "val",
        args.batch_size,
        mean,
        std,
        args.workers,
        args.seed + 2,
        keep=reco_clean_keep,
    )
    internal_val = loader(
        args.internal_pred_data,
        "val",
        args.batch_size,
        mean,
        std,
        args.workers,
        args.seed + 3,
        keep=pred_keep,
    )
    gt_val = loader(
        args.gt_data, "val", args.batch_size, mean, std, args.workers, args.seed + 4, keep=gt_keep
    )

    def validate(step):
        result = {
            "step": step,
            "reco_val": evaluate(model, reco_val, device),
            "reco_val_primary_unseen": evaluate(model, reco_clean_val, device),
            "internal_pred_val": evaluate(model, internal_val, device),
            "gt_clean_val": evaluate(model, gt_val, device),
        }
        (args.output_dir / f"val_step_{step}.json").write_text(json.dumps(result, indent=2) + "\n")
        print("validation " + json.dumps(result, sort_keys=True), flush=True)
        return result

    initial = validate(0)
    best_clean_reco_q68 = initial["reco_val_primary_unseen"]["q68"]
    for step in range(1, args.steps + 1):
        model.train()
        x, target, mask, _particle = next(reco_train)
        x, target, mask = x.to(device), target.to(device), mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            direction, _point = model(x, mask)
            loss = loss_function(direction, target, "trimmed_angular", 1.0, None, 12.0)
        loss.backward()
        if final.weight.grad is not None:
            final.weight.grad[2:].zero_()
        if final.bias.grad is not None:
            final.bias.grad[2:].zero_()
        torch.nn.utils.clip_grad_norm_(final.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            final.weight[2:].copy_(point_weight_initial)
            final.bias[2:].copy_(point_bias_initial)
        if step % 100 == 0:
            print(f"step={step} loss={float(loss.detach()):.6f}", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            current = validate(step)
            admissible = (
                initial["reco_val_primary_unseen"]["count"] >= args.min_clean_reco_val
                and current["reco_val_primary_unseen"]["q68"]
                <= initial["reco_val_primary_unseen"]["q68"] - 0.05
                and current["reco_val_primary_unseen"]["q50"]
                <= initial["reco_val_primary_unseen"]["q50"] + 0.05
                and current["internal_pred_val"]["q68"]
                <= initial["internal_pred_val"]["q68"] + 0.05
                and current["gt_clean_val"]["q68"] <= initial["gt_clean_val"]["q68"] + 0.05
                and current["reco_val"]["q50"] <= initial["reco_val"]["q50"] + 0.05
            )
            print(f"gate step={step} admissible={admissible}", flush=True)
            if (
                admissible
                and current["reco_val_primary_unseen"]["q68"] < best_clean_reco_q68 - 0.05
            ):
                best_clean_reco_q68 = current["reco_val_primary_unseen"]["q68"]
                torch.save(
                    {"model": model.state_dict(), "args": vars(args), "metrics": current},
                    args.output_dir / "best.pt",
                )
    print(f"best_clean_reco_q68={best_clean_reco_q68:.6f}", flush=True)


if __name__ == "__main__":
    main()
