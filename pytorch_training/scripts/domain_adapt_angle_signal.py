#!/usr/bin/env python3
"""Accuracy-constrained angle-domain adaptation on matched signal selections.

The initial model is trained with truth-signal MC.  Fine-tuning combines:
  * supervised all-particle truth-signal MC loss,
  * supervised predicted-signal atmospheric-muon MC loss, and
  * sliced-Wasserstein output plus CORAL feature alignment between the same
    predicted-signal MC selection and unlabeled experimental events.

Checkpoints selected for alignment are only accepted while validation q68 stays
within a configurable factor of the initial reconstruction checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_angle_signal import (  # noqa: E402
    BatchedAngleDataset,
    build_model,
    evaluate,
    infinite_batches,
    loss_function,
    make_loader,
    seed_everything,
)


class UnlabeledAngleDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str,
        events_per_batch: int,
        max_seq_len: int,
        center_time: bool,
        max_events: int | None = None,
    ) -> None:
        self.path = path
        self.split = split
        self.events_per_batch = events_per_batch
        self.max_seq_len = max_seq_len
        self.center_time = center_time
        self._file = None
        with h5py.File(path, "r") as h5:
            self.n_events = len(h5[f"{split}/ev_starts/data"]) - 1
            if max_events is not None:
                self.n_events = min(self.n_events, max_events)
            self.feature_mean = np.asarray(h5["norm_param/mean"], dtype=np.float32)
            self.feature_std = np.asarray(h5["norm_param/std"], dtype=np.float32)

    @property
    def h5(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r", swmr=True)
        return self._file

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __len__(self) -> int:
        return math.ceil(self.n_events / self.events_per_batch)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        event_start = index * self.events_per_batch
        event_end = min(event_start + self.events_per_batch, self.n_events)
        starts = np.asarray(
            self.h5[f"{self.split}/ev_starts/data"][event_start : event_end + 1],
            dtype=np.int64,
        )
        hit_start, hit_end = int(starts[0]), int(starts[-1])
        features = np.asarray(
            self.h5[f"{self.split}/data/data"][hit_start:hit_end], dtype=np.float32
        )
        rel = starts - hit_start
        lengths = np.minimum(np.diff(rel), self.max_seq_len).astype(np.int64)
        seq_len = int(lengths.max())
        x = np.zeros((event_end - event_start, seq_len, 5), dtype=np.float32)
        mask = np.zeros((event_end - event_start, seq_len), dtype=bool)
        for row, length in enumerate(lengths):
            values = features[int(rel[row]) : int(rel[row]) + length].copy()
            if self.center_time:
                values[:, 1] -= values[:, 1].mean()
            x[row, :length] = values
            mask[row, :length] = True
        return torch.from_numpy(x), torch.from_numpy(mask)


def sliced_wasserstein(a: Tensor, b: Tensor, projections: Tensor) -> Tensor:
    n = min(len(a), len(b))
    a = a[:n] @ projections.T
    b = b[:n] @ projections.T
    return torch.abs(torch.sort(a, dim=0).values - torch.sort(b, dim=0).values).mean()


def coral_loss(a: Tensor, b: Tensor) -> Tensor:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    a = (a - a.mean(0)) / a.std(0).clamp_min(1e-3)
    b = (b - b.mean(0)) / b.std(0).clamp_min(1e-3)
    mean_loss = torch.square(a.mean(0) - b.mean(0)).mean()
    a = a - a.mean(0)
    b = b - b.mean(0)
    cov_a = a.T @ a / max(n - 1, 1)
    cov_b = b.T @ b / max(n - 1, 1)
    return mean_loss + torch.square(cov_a - cov_b).mean()


@torch.inference_mode()
def predict_labeled(model, loader, device: torch.device) -> np.ndarray:
    model.eval()
    parts = []
    for x, _, mask, _ in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            parts.append(model(x, mask).float().cpu().numpy())
    return np.concatenate(parts)


@torch.inference_mode()
def predict_unlabeled(model, loader, device: torch.device) -> np.ndarray:
    model.eval()
    parts = []
    for x, mask in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            parts.append(model(x, mask).float().cpu().numpy())
    return np.concatenate(parts)


def empirical_wasserstein(a: np.ndarray, b: np.ndarray, circular: bool = False) -> float:
    quantiles = np.linspace(0.0, 1.0, 2001)
    qa, qb = np.quantile(a, quantiles), np.quantile(b, quantiles)
    delta = np.abs(qa - qb)
    if circular:
        delta = np.minimum(delta, 360.0 - delta)
    return float(delta.mean())


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    values = np.sort(np.concatenate((a, b)))
    cdf_a = np.searchsorted(np.sort(a), values, side="right") / len(a)
    cdf_b = np.searchsorted(np.sort(b), values, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def js_divergence_2d(theta_a, phi_a, theta_b, phi_b) -> float:
    bins = (np.linspace(0, 180, 19), np.linspace(0, 360, 37))
    pa = np.histogram2d(theta_a, phi_a, bins=bins)[0].ravel().astype(np.float64) + 1e-12
    pb = np.histogram2d(theta_b, phi_b, bins=bins)[0].ravel().astype(np.float64) + 1e-12
    pa /= pa.sum()
    pb /= pb.sum()
    midpoint = 0.5 * (pa + pb)
    return float(
        0.5 * np.sum(pa * np.log(pa / midpoint)) + 0.5 * np.sum(pb * np.log(pb / midpoint))
    )


def numpy_direction_swd(a: np.ndarray, b: np.ndarray, seed: int = 123) -> float:
    rng = np.random.default_rng(seed)
    projections = rng.normal(size=(64, 3))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)
    quantiles = np.linspace(0.0, 1.0, 1001)
    distances = []
    for projection in projections:
        distances.append(
            np.abs(
                np.quantile(a @ projection, quantiles) - np.quantile(b @ projection, quantiles)
            ).mean()
        )
    return float(np.mean(distances))


def alignment_metrics(mc: np.ndarray, exp: np.ndarray) -> dict[str, float]:
    theta_mc = np.degrees(np.arccos(np.clip(mc[:, 2], -1, 1)))
    theta_exp = np.degrees(np.arccos(np.clip(exp[:, 2], -1, 1)))
    phi_mc = np.mod(np.degrees(np.arctan2(mc[:, 1], mc[:, 0])), 360.0)
    phi_exp = np.mod(np.degrees(np.arctan2(exp[:, 1], exp[:, 0])), 360.0)
    return {
        "mc_count": int(len(mc)),
        "exp_count": int(len(exp)),
        "direction_swd": numpy_direction_swd(mc, exp),
        "theta_wasserstein_deg": empirical_wasserstein(theta_mc, theta_exp),
        "theta_ks": ks_statistic(theta_mc, theta_exp),
        "phi_wasserstein_deg": empirical_wasserstein(phi_mc, phi_exp, circular=True),
        "theta_phi_js": js_divergence_2d(theta_mc, phi_mc, theta_exp, phi_exp),
        "mc_theta_mean": float(theta_mc.mean()),
        "exp_theta_mean": float(theta_exp.mean()),
        "mc_theta_std": float(theta_mc.std()),
        "exp_theta_std": float(theta_exp.std()),
        "mc_phi_resultant": float(
            np.hypot(np.cos(np.deg2rad(phi_mc)).mean(), np.sin(np.deg2rad(phi_mc)).mean())
        ),
        "exp_phi_resultant": float(
            np.hypot(np.cos(np.deg2rad(phi_exp)).mean(), np.sin(np.deg2rad(phi_exp)).mean())
        ),
    }


def validate(model, truth_loader, mc_loader, exp_loader, device) -> dict:
    truth = evaluate(model, truth_loader, device)
    matched = evaluate(model, mc_loader, device)
    mc_pred = predict_labeled(model, mc_loader, device)
    exp_pred = predict_unlabeled(model, exp_loader, device)
    return {
        "truth": truth,
        "matched_mc": matched,
        "alignment": alignment_metrics(mc_pred, exp_pred),
    }


def save_checkpoint(path: Path, model, optimizer, step: int, args, init_args, metrics) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "args": vars(args),
            "initial_model_args": vars(init_args),
            "metrics": metrics,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--truth-data", required=True)
    parser.add_argument("--matched-mc-data", required=True)
    parser.add_argument("--experimental-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--truth-batch-size", type=int, default=512)
    parser.add_argument("--domain-batch-size", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-truth-val-events", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--truth-loss-weight", type=float, default=1.0)
    parser.add_argument("--matched-loss-weight", type=float, default=1.0)
    parser.add_argument("--swd-weight", type=float, default=0.1)
    parser.add_argument("--coral-weight", type=float, default=0.01)
    parser.add_argument("--q68-constraint-factor", type=float, default=1.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=52)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.initial_checkpoint, map_location="cpu")
    init_args = SimpleNamespace(**checkpoint["args"])
    model = build_model(init_args).to(device)
    model.load_state_dict(checkpoint["model"])
    center_time = bool(getattr(init_args, "center_time", False))

    truth_train = BatchedAngleDataset(
        args.truth_data,
        "train",
        args.truth_batch_size,
        args.max_seq_len,
        bool(getattr(init_args, "use_signal_norm", False)),
        None,
        center_time,
    )
    truth_val = BatchedAngleDataset(
        args.truth_data,
        "val",
        args.truth_batch_size,
        args.max_seq_len,
        bool(getattr(init_args, "use_signal_norm", False)),
        args.max_truth_val_events,
        center_time,
    )
    mc_train = BatchedAngleDataset(
        args.matched_mc_data,
        "train",
        args.domain_batch_size,
        args.max_seq_len,
        False,
        None,
        center_time,
    )
    mc_val = BatchedAngleDataset(
        args.matched_mc_data,
        "val",
        args.domain_batch_size,
        args.max_seq_len,
        False,
        None,
        center_time,
    )
    exp_train = UnlabeledAngleDataset(
        args.experimental_data,
        "train",
        args.domain_batch_size,
        args.max_seq_len,
        center_time,
    )
    exp_val = UnlabeledAngleDataset(
        args.experimental_data,
        "val",
        args.domain_batch_size,
        args.max_seq_len,
        center_time,
    )
    truth_train_it = infinite_batches(make_loader(truth_train, args.workers, True, args.seed))
    mc_train_it = infinite_batches(make_loader(mc_train, args.workers, True, args.seed + 1))
    exp_train_it = infinite_batches(make_loader(exp_train, args.workers, True, args.seed + 2))
    truth_val_loader = make_loader(truth_val, args.workers, False, args.seed)
    mc_val_loader = make_loader(mc_val, args.workers, False, args.seed)
    exp_val_loader = make_loader(exp_val, args.workers, False, args.seed)

    baseline = validate(model, truth_val_loader, mc_val_loader, exp_val_loader, device)
    (output_dir / "baseline_metrics.json").write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n"
    )
    print("baseline " + json.dumps(baseline, sort_keys=True), flush=True)
    truth_limit = baseline["truth"]["q68"] * args.q68_constraint_factor
    matched_limit = baseline["matched_mc"]["q68"] * args.q68_constraint_factor

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=device.type == "cuda"
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 100)
    projections = torch.randn((64, 3), generator=generator, device=device)
    projections = torch.nn.functional.normalize(projections, dim=1)
    particle_loss_weights = torch.as_tensor(
        getattr(init_args, "particle_loss_weights", (1.0, 1.0, 1.0)),
        device=device,
    )
    best_alignment = baseline["alignment"]["direction_swd"]
    best_truth_q68 = baseline["truth"]["q68"]
    running = []
    log_time = time.time()

    for step in range(1, args.max_steps + 1):
        model.train()
        truth_x, truth_y, truth_mask, truth_particle = next(truth_train_it)
        mc_x, mc_y, mc_mask, _ = next(mc_train_it)
        exp_x, exp_mask = next(exp_train_it)
        truth_x, truth_y, truth_mask = (
            truth_x.to(device, non_blocking=True),
            truth_y.to(device, non_blocking=True),
            truth_mask.to(device, non_blocking=True),
        )
        truth_particle = truth_particle.to(device, non_blocking=True)
        mc_x, mc_y, mc_mask = (
            mc_x.to(device, non_blocking=True),
            mc_y.to(device, non_blocking=True),
            mc_mask.to(device, non_blocking=True),
        )
        exp_x, exp_mask = (
            exp_x.to(device, non_blocking=True),
            exp_mask.to(device, non_blocking=True),
        )
        progress = step / args.max_steps
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            truth_pred = model(truth_x, truth_mask)
            mc_pred, mc_features = model(mc_x, mc_mask, return_features=True)
            exp_pred, exp_features = model(exp_x, exp_mask, return_features=True)
            truth_event_weights = particle_loss_weights[truth_particle.clamp_min(0)]
            truth_loss = loss_function(
                truth_pred,
                truth_y,
                init_args.loss,
                init_args.loss_power,
                truth_event_weights,
            )
            matched_loss = loss_function(mc_pred, mc_y, init_args.loss, init_args.loss_power)
            swd = sliced_wasserstein(mc_pred, exp_pred, projections)
            coral = coral_loss(mc_features.float(), exp_features.float())
            loss = (
                args.truth_loss_weight * truth_loss
                + args.matched_loss_weight * matched_loss
                + args.swd_weight * swd
                + args.coral_weight * coral
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        running.append(
            (
                float(loss.detach()),
                float(truth_loss.detach()),
                float(matched_loss.detach()),
                float(swd.detach()),
                float(coral.detach()),
            )
        )

        if step % args.log_every == 0:
            values = np.mean(running, axis=0)
            print(
                f"step={step} total={values[0]:.6f} truth={values[1]:.6f} "
                f"matched={values[2]:.6f} swd={values[3]:.6f} coral={values[4]:.6f} "
                f"lr={lr:.3e} steps/s={len(running) / (time.time() - log_time):.2f}",
                flush=True,
            )
            running.clear()
            log_time = time.time()

        if step % args.val_every == 0 or step == args.max_steps:
            metrics = validate(model, truth_val_loader, mc_val_loader, exp_val_loader, device)
            metrics["step"] = step
            metrics["feasible"] = bool(
                metrics["truth"]["q68"] <= truth_limit
                and metrics["matched_mc"]["q68"] <= matched_limit
            )
            print("validation " + json.dumps(metrics, sort_keys=True), flush=True)
            (output_dir / "last_metrics.json").write_text(
                json.dumps(metrics, indent=2, sort_keys=True) + "\n"
            )
            save_checkpoint(
                output_dir / "last.pt", model, optimizer, step, args, init_args, metrics
            )
            if metrics["truth"]["q68"] < best_truth_q68:
                best_truth_q68 = metrics["truth"]["q68"]
                save_checkpoint(
                    output_dir / "best_reco.pt", model, optimizer, step, args, init_args, metrics
                )
            alignment = metrics["alignment"]["direction_swd"]
            if metrics["feasible"] and alignment < best_alignment:
                best_alignment = alignment
                save_checkpoint(
                    output_dir / "best_alignment.pt",
                    model,
                    optimizer,
                    step,
                    args,
                    init_args,
                    metrics,
                )


if __name__ == "__main__":
    main()
