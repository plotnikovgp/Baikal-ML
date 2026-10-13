#!/usr/bin/env python3
"""Fine-tune one angle/point model on predicted hits and nue2 GT hits.

Optional single-muatm filtering removes events with multiple simulated muons
from predicted-hit training and validation. The model is initialized from the
nue2 angle/point specialist. Cross-dataset masks keep evaluation independent
of both that specialist and the older all-particle predicted-hit model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from physics_feature_upgrade import upgrade_track_model_physics_features
from physics_graph_upgrade import upgrade_track_model_graph
from softscore_model import upgrade_track_model
from span_crop_augmentation import crop_vertical_span
from torch import nn
from track_augmentation import augment_track_batch
from train_angle_signal import (
    BatchedAngleDataset,
    SetTransformerDirection,
    infinite_batches,
    loss_function,
    make_loader,
    seed_everything,
)
from train_track_anchor_v3 import BatchedTrackDataset, DirectionTrackAnchor, point_loss
from train_track_uncertainty_v2 import residuals


def thin_predicted_hits(
    x: torch.Tensor,
    mask: torch.Tensor,
    probabilities: torch.Tensor,
    direction: torch.Tensor,
    point: torch.Tensor,
    valid: torch.Tensor,
    original_centroid_m: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    probability: float,
    threshold_min: float,
    threshold_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vary the signal threshold while preserving the true physical track line."""
    if not len(x) or probability <= 0:
        return x, mask, point
    apply = torch.rand(len(x)) < probability
    thresholds = torch.empty(len(x)).uniform_(threshold_min, threshold_max)
    candidate = mask & (probabilities >= thresholds[:, None])
    # Keep the original cut: at least eight hits on at least two strings.
    enough_hits = candidate.sum(dim=1) >= 8
    positions_m = x[..., 2:5] * std[2:5] + mean[2:5]
    first_index = candidate.int().argmax(dim=1)
    first_xy = positions_m[torch.arange(len(x)), first_index, :2]
    distinct_string = (
        (torch.linalg.vector_norm(positions_m[..., :2] - first_xy[:, None], dim=-1) > 1.0)
        & candidate
    ).any(dim=1)
    apply &= enough_hits & distinct_string
    new_mask = torch.where(apply[:, None], candidate, mask)
    changed = apply & (new_mask != mask).any(dim=1) & valid
    if changed.any():
        count = new_mask.sum(dim=1).clamp_min(1)[:, None]
        new_centroid_m = (positions_m * new_mask[..., None]).sum(dim=1) / count
        rebased = point + original_centroid_m - new_centroid_m
        rebased -= direction * (rebased * direction).sum(dim=1, keepdim=True)
        point = torch.where(changed[:, None], rebased, point)
    return x * new_mask[..., None], new_mask, point


class AdaptedTrackDataset(BatchedTrackDataset):
    def __init__(
        self,
        *args,
        keep: np.ndarray | None,
        expected_mean,
        expected_std,
        single_muatm_only: bool = False,
        pred_thinning_prob: float = 0.0,
        pred_threshold_min: float = 0.70,
        pred_threshold_max: float = 0.90,
        span_crop_prob: float = 0.0,
        span_crop_min_keep: float = 0.75,
        span_crop_max_keep: float = 0.90,
        score_feature: bool = False,
        score_jitter: float = 0.0,
        rotate_z_prob: float = 0.0,
        time_jitter_ns: float = 0.0,
        event_time_jitter_ns: float = 0.0,
        coord_jitter_m: float = 0.0,
        charge_log_jitter: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.keep = keep
        self.single_muatm_only = single_muatm_only
        self.pred_thinning_prob = pred_thinning_prob
        self.pred_threshold_min = pred_threshold_min
        self.pred_threshold_max = pred_threshold_max
        self.span_crop_prob = span_crop_prob
        self.span_crop_min_keep = span_crop_min_keep
        self.span_crop_max_keep = span_crop_max_keep
        self.score_feature = score_feature
        self.score_jitter = score_jitter
        self.rotate_z_prob = rotate_z_prob
        self.time_jitter_ns = time_jitter_ns
        self.event_time_jitter_ns = event_time_jitter_ns
        self.coord_jitter_m = coord_jitter_m
        self.charge_log_jitter = charge_log_jitter
        with h5py.File(self.path, "r") as h5:
            self.has_signal_probability = f"{self.split}/signal_probability/data" in h5
        if pred_thinning_prob > 0 and not self.has_signal_probability:
            raise ValueError("Predicted-hit thinning requires signal_probability/data")
        if keep is not None and len(keep) < self.n_events:
            raise ValueError(f"Selection mask too short for {self.split}")
        self.source_mean = torch.tensor(self.target_mean, dtype=torch.float32)
        self.source_std = torch.tensor(self.target_std, dtype=torch.float32)
        self.expected_mean = torch.tensor(expected_mean, dtype=torch.float32)
        self.expected_std = torch.tensor(expected_std, dtype=torch.float32)

    def __getitem__(self, batch_index: int):
        x, direction, mask, particle, point, valid = super().__getitem__(batch_index)
        x = (x * self.source_std + self.source_mean - self.expected_mean) / self.expected_std
        x = x * mask.unsqueeze(-1)
        event_start = batch_index * self.events_per_batch
        event_end = event_start + len(x)
        selection = torch.ones(len(x), dtype=torch.bool)
        if self.keep is not None:
            selection &= torch.from_numpy(self.keep[event_start:event_end].copy())
        if self.single_muatm_only:
            selection &= (particle != 0) | valid
        if (self.pred_thinning_prob > 0 or self.score_feature) and selection.any():
            prob = np.ones(mask.shape, dtype=np.float32)
        if (
            self.has_signal_probability
            and (self.pred_thinning_prob > 0 or self.score_feature)
            and selection.any()
        ):
            starts = np.asarray(
                self.h5[f"{self.split}/ev_starts/data"][event_start : event_end + 1],
                dtype=np.int64,
            )
            flat_prob = np.asarray(
                self.h5[f"{self.split}/signal_probability/data"][starts[0] : starts[-1]],
                dtype=np.float32,
            )
            for i, length in enumerate(mask.sum(dim=1).tolist()):
                prob[i, :length] = flat_prob[starts[i] - starts[0] : starts[i] - starts[0] + length]
        if self.pred_thinning_prob > 0 and selection.any():
            original_centroid_m = torch.from_numpy(
                np.asarray(
                    self.anchor_h5[f"{self.split}/hit_centroid_m"][event_start:event_end],
                    dtype=np.float32,
                )
            )
        if not torch.all(selection):
            x, direction, mask, particle, point, valid = (
                value[selection] for value in (x, direction, mask, particle, point, valid)
            )
            if (self.pred_thinning_prob > 0 or self.score_feature) and selection.any():
                prob = prob[selection]
            if self.pred_thinning_prob > 0 and selection.any():
                original_centroid_m = original_centroid_m[selection]
        if self.pred_thinning_prob > 0 and len(x):
            x, mask, point = thin_predicted_hits(
                x,
                mask,
                torch.from_numpy(prob),
                direction,
                point,
                valid,
                original_centroid_m,
                self.expected_mean,
                self.expected_std,
                self.pred_thinning_prob,
                self.pred_threshold_min,
                self.pred_threshold_max,
            )
        if self.span_crop_prob > 0 and len(x):
            x, mask, point = crop_vertical_span(
                x,
                mask,
                direction,
                point,
                valid,
                self.expected_mean,
                self.expected_std,
                probability=self.span_crop_prob,
                min_keep_fraction=self.span_crop_min_keep,
                max_keep_fraction=self.span_crop_max_keep,
            )
        if self.score_feature:
            if len(x):
                score = torch.from_numpy(prob)
                if self.split == "train" and self.has_signal_probability and self.score_jitter:
                    score = (score + torch.randn_like(score) * self.score_jitter).clamp(0.7, 1.0)
                score = score * mask
            else:
                score = torch.zeros(mask.shape, dtype=torch.float32)
            x = torch.cat((x, score.unsqueeze(-1)), dim=-1)
        if self.split == "train":
            x, direction, point = augment_track_batch(
                x,
                direction,
                mask,
                point,
                self.expected_mean,
                self.expected_std,
                rotate_z_prob=self.rotate_z_prob,
                time_jitter_ns=self.time_jitter_ns,
                event_time_jitter_ns=self.event_time_jitter_ns,
                coord_jitter_m=self.coord_jitter_m,
                charge_log_jitter=self.charge_log_jitter,
            )
        return x, direction, mask, particle, point, valid


class CleanParticleAngleDataset(BatchedAngleDataset):
    """Angle-only clean-hit stream, normalized like the shared model input."""

    def __init__(
        self,
        *args,
        expected_mean: np.ndarray,
        expected_std: np.ndarray,
        particle_code: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.source_mean = torch.tensor(self.target_mean, dtype=torch.float32)
        self.source_std = torch.tensor(self.target_std, dtype=torch.float32)
        self.expected_mean = torch.tensor(expected_mean, dtype=torch.float32)
        self.expected_std = torch.tensor(expected_std, dtype=torch.float32)
        self.particle_code = particle_code

    def __getitem__(self, batch_index: int):
        x, direction, mask, particle = super().__getitem__(batch_index)
        x = (x * self.source_std + self.source_mean - self.expected_mean) / self.expected_std
        selected = particle == self.particle_code
        return (
            (x * mask.unsqueeze(-1))[selected],
            direction[selected],
            mask[selected],
            particle[selected],
        )


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "q50": float(np.quantile(values, 0.5)),
        "q68": float(np.quantile(values, 0.68)),
        "q95": float(np.quantile(values, 0.95)),
    }


def next_nonempty(source):
    while True:
        batch = next(source)
        if len(batch[0]):
            return batch


def validation_score(result: dict, pred_weight: float, muatm_weight: float) -> float:
    pred = result["pred_val"]
    gt = result["gt_val_clean"]
    return (
        gt["direction_deg"]["q68"]
        + 0.03 * gt["line_m"]["q68"]
        + pred_weight * pred["direction_deg"]["q68"]
        + muatm_weight * pred["muatm"]["direction_deg"]["q68"]
    )


@torch.inference_mode()
def evaluate(model: nn.Module, loader, device: torch.device, *, return_arrays: bool = False):
    model.eval()
    chunks = {name: [] for name in ("angle", "line", "valid", "particle")}
    for x, target_dir, mask, particle, target_point, valid in loader:
        if not len(x):
            continue
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        target_dir = target_dir.to(device, non_blocking=True)
        target_point = target_point.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            direction, point = model(x, mask)
        angle, line, _ = residuals(direction.float(), point.float(), target_dir, target_point)
        chunks["angle"].append(angle.cpu().numpy())
        chunks["line"].append(line.cpu().numpy())
        chunks["valid"].append(valid.numpy())
        chunks["particle"].append(particle.numpy())
    pred = {key: np.concatenate(value) for key, value in chunks.items()}
    result = {
        "count": int(len(pred["angle"])),
        "track_count": int(pred["valid"].sum()),
        "direction_deg": quantiles(pred["angle"]),
        "line_m": quantiles(pred["line"][pred["valid"]]),
    }
    for code, name in ((0, "muatm"), (1, "nuatm"), (2, "nue2")):
        selected = pred["particle"] == code
        if selected.any():
            track = selected & pred["valid"]
            result[name] = {
                "count": int(selected.sum()),
                "track_count": int(track.sum()),
                "direction_deg": quantiles(pred["angle"][selected]),
            }
            if track.any():
                result[name]["line_m"] = quantiles(pred["line"][track])
    return (result, pred) if return_arrays else result


@torch.inference_mode()
def evaluate_angle_only(model: nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    errors = []
    for x, target_dir, mask, _particle in loader:
        if not len(x):
            continue
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            direction, _point = model(x.to(device), mask.to(device))
        dot = (direction.float() * target_dir.to(device)).sum(dim=1).clamp(-1.0, 1.0)
        errors.append(torch.rad2deg(torch.acos(dot)).cpu().numpy())
    values = np.concatenate(errors)
    return {"count": int(len(values)), "direction_deg": quantiles(values)}


def model_from_checkpoints(angle_path: Path, track_path: Path, scale: float, device):
    angle = torch.load(angle_path, map_location="cpu", weights_only=False)
    config = angle["args"]
    backbone = SetTransformerDirection(
        config["hidden"],
        config["layers"],
        config["heads"],
        config["ff_mult"],
        config["dropout"],
        config["feature_mode"],
    )
    model = DirectionTrackAnchor(backbone, config["hidden"], config["dropout"], scale)
    track = torch.load(track_path, map_location="cpu", weights_only=False)
    model.load_state_dict(track["model"], strict=True)
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-data", type=Path, required=True)
    parser.add_argument("--norm-data", type=Path, default=None)
    parser.add_argument("--pred-anchors", type=Path, required=True)
    parser.add_argument("--gt-data", type=Path, required=True)
    parser.add_argument("--gt-anchors", type=Path, required=True)
    parser.add_argument("--clean-nuatm-data", type=Path, default=None)
    parser.add_argument("--clean-nuatm-weight", type=float, default=0.0)
    parser.add_argument("--clean-paired-data", type=Path, default=None)
    parser.add_argument("--clean-paired-anchors", type=Path, default=None)
    parser.add_argument("--clean-paired-weight", type=float, default=0.0)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--angle-checkpoint", type=Path, required=True)
    parser.add_argument("--track-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--gt-weight", type=float, default=0.5)
    parser.add_argument("--point-weight", type=float, default=0.5)
    parser.add_argument(
        "--pred-direction-loss",
        choices=("mae", "mse", "cosine", "chord", "trimmed_angular", "angular_huber"),
        default="mae",
    )
    parser.add_argument("--pred-direction-power", type=float, default=1.0)
    parser.add_argument("--pred-direction-cap-deg", type=float, default=8.0)
    parser.add_argument("--pred-direction-scale", type=float, default=1.0)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--point-lr", type=float, default=1e-4)
    parser.add_argument("--direction-head-only", action="store_true")
    parser.add_argument("--teacher-weight", type=float, default=0.0)
    parser.add_argument("--muatm-teacher-angle", type=Path, default=None)
    parser.add_argument("--muatm-teacher-track", type=Path, default=None)
    parser.add_argument("--max-pred-q68", type=float, default=None)
    parser.add_argument("--max-gt-q68", type=float, default=None)
    parser.add_argument("--pred-score-weight", type=float, default=0.0)
    parser.add_argument("--muatm-score-weight", type=float, default=0.0)
    parser.add_argument("--single-muatm-only", action="store_true")
    parser.add_argument("--pred-thinning-prob", type=float, default=0.0)
    parser.add_argument("--pred-threshold-min", type=float, default=0.70)
    parser.add_argument("--pred-threshold-max", type=float, default=0.90)
    parser.add_argument("--span-crop-prob", type=float, default=0.0)
    parser.add_argument("--span-crop-min-keep", type=float, default=0.75)
    parser.add_argument("--span-crop-max-keep", type=float, default=0.90)
    parser.add_argument("--score-feature", action="store_true")
    parser.add_argument("--score-jitter", type=float, default=0.0)
    parser.add_argument("--rotate-z-prob", type=float, default=0.0)
    parser.add_argument("--time-jitter-ns", type=float, default=0.0)
    parser.add_argument("--event-time-jitter-ns", type=float, default=0.0)
    parser.add_argument("--coord-jitter-m", type=float, default=0.0)
    parser.add_argument("--charge-log-jitter", type=float, default=0.0)
    parser.add_argument("--score-mode", choices=("absolute", "centered"), default="absolute")
    parser.add_argument("--init-from-five-feature-model", action="store_true")
    parser.add_argument("--score-column-only", action="store_true")
    parser.add_argument("--score-column-lr", type=float, default=1e-3)
    parser.add_argument("--graph-layers", type=int, default=0)
    parser.add_argument("--graph-k", type=int, default=12)
    parser.add_argument("--graph-photon-speed-m-per-ns", type=float, default=0.225)
    parser.add_argument("--graph-minkowski-edge", action="store_true")
    parser.add_argument("--graph-only", action="store_true")
    parser.add_argument("--graph-lr", type=float, default=1e-4)
    parser.add_argument("--physics-features", action="store_true")
    parser.add_argument("--physics-input-lr", type=float, default=1e-4)
    parser.add_argument("--init-from-transformer-model", action="store_true")
    parser.add_argument("--anchor-scale-m", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-split", choices=("val", "test"), default="val")
    parser.add_argument("--load-model", type=Path, default=None)
    args = parser.parse_args()
    if not 0 <= args.gt_weight <= 1:
        raise ValueError("gt-weight must be between 0 and 1")
    if args.clean_nuatm_weight < 0 or (
        args.clean_nuatm_weight > 0 and args.clean_nuatm_data is None
    ):
        raise ValueError("Positive clean-nuatm weight needs --clean-nuatm-data")
    if args.clean_paired_weight < 0 or (
        args.clean_paired_weight > 0
        and (args.clean_paired_data is None or args.clean_paired_anchors is None)
    ):
        raise ValueError("Positive clean-paired weight needs data and anchors")
    if args.clean_nuatm_weight > 0 and args.clean_paired_weight > 0:
        raise ValueError("Choose one clean-hit augmentation stream at a time")
    if args.pred_direction_scale <= 0 or args.pred_direction_power <= 0:
        raise ValueError("Direction-loss scale and power must be positive")
    if not 0 <= args.pred_thinning_prob <= 1:
        raise ValueError("pred-thinning-prob must be between 0 and 1")
    if not 0.70 <= args.pred_threshold_min <= args.pred_threshold_max <= 1:
        raise ValueError("Invalid predicted-hit threshold interval")
    if not 0 <= args.span_crop_prob <= 1:
        raise ValueError("span-crop-prob must be in [0, 1]")
    if not 0 < args.span_crop_min_keep <= args.span_crop_max_keep <= 1:
        raise ValueError("Invalid span-crop kept-hit fraction interval")
    if args.score_jitter < 0:
        raise ValueError("Score jitter must be nonnegative")
    if not 0 <= args.rotate_z_prob <= 1:
        raise ValueError("rotate-z-prob must be in [0, 1]")
    if (
        min(
            args.time_jitter_ns,
            args.event_time_jitter_ns,
            args.coord_jitter_m,
            args.charge_log_jitter,
        )
        < 0
    ):
        raise ValueError("Physical jitter magnitudes must be nonnegative")
    if args.init_from_five_feature_model and (not args.score_feature or args.load_model is None):
        raise ValueError("Five-feature initialization needs --score-feature and --load-model")
    if args.score_feature and (args.direction_head_only or args.teacher_weight > 0):
        raise ValueError("Score-feature training needs the full backbone and no legacy teacher")
    if args.score_column_only and (not args.score_feature or args.score_column_lr <= 0):
        raise ValueError(
            "Score-column-only training needs a score feature and positive learning rate"
        )
    if args.graph_layers < 0 or args.graph_k < 1 or args.graph_photon_speed_m_per_ns <= 0:
        raise ValueError("Invalid graph geometry settings")
    if args.graph_layers and args.score_feature:
        raise ValueError("Graph and score-feature upgrades are separate experiments")
    if args.physics_features and (
        args.graph_layers or args.score_feature or args.direction_head_only
    ):
        raise ValueError("Physical input features require the plain Transformer")
    if args.physics_features and args.physics_input_lr <= 0:
        raise ValueError("Physical input learning rate must be positive")
    if args.graph_only and (args.graph_layers < 1 or args.graph_lr <= 0):
        raise ValueError("Graph-only training needs graph layers and positive graph LR")
    if args.init_from_transformer_model and (
        (args.graph_layers < 1 and not args.physics_features) or args.load_model is None
    ):
        raise ValueError(
            "Transformer initialization needs an architecture upgrade and --load-model"
        )
    if args.init_from_five_feature_model and args.init_from_transformer_model:
        raise ValueError("Select only one architecture initialization mode")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n"
    )
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_from_checkpoints(
        args.angle_checkpoint, args.track_checkpoint, args.anchor_scale_m, device
    )
    if args.load_model is not None and (
        args.init_from_five_feature_model or args.init_from_transformer_model
    ):
        checkpoint = torch.load(args.load_model, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
    if args.score_feature:
        model = upgrade_track_model(model, args.score_mode).to(device)
    if args.graph_layers:
        model = upgrade_track_model_graph(
            model,
            graph_layers=args.graph_layers,
            graph_k=args.graph_k,
            photon_speed_m_per_ns=args.graph_photon_speed_m_per_ns,
            minkowski_edge=args.graph_minkowski_edge,
        ).to(device)
    if args.physics_features:
        model = upgrade_track_model_physics_features(model).to(device)
    if args.load_model is not None and not (
        args.init_from_five_feature_model or args.init_from_transformer_model
    ):
        checkpoint = torch.load(args.load_model, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
    teacher = None
    muatm_teacher = None
    if args.teacher_weight > 0 and not args.eval_only:
        teacher = model_from_checkpoints(
            args.angle_checkpoint, args.track_checkpoint, args.anchor_scale_m, device
        )
        teacher.eval().requires_grad_(False)
        if (args.muatm_teacher_angle is None) != (args.muatm_teacher_track is None):
            raise ValueError("Both muatm teacher checkpoints must be specified together")
        if args.muatm_teacher_angle is not None:
            muatm_teacher = model_from_checkpoints(
                args.muatm_teacher_angle,
                args.muatm_teacher_track,
                args.anchor_scale_m,
                device,
            )
            muatm_teacher.eval().requires_grad_(False)
    with h5py.File(args.norm_data or args.pred_data, "r") as h5:
        expected_mean = np.asarray(h5["norm_param/mean"], dtype=np.float32)
        expected_std = np.asarray(h5["norm_param/std"], dtype=np.float32)
    if args.graph_layers or args.physics_features:
        model.backbone.set_feature_norm(expected_mean, expected_std)
    with h5py.File(args.pred_data, "r") as h5:
        pred_mean = np.asarray(h5["norm_param/mean"], dtype=np.float32)
        pred_std = np.asarray(h5["norm_param/std"], dtype=np.float32)
    student_mean = torch.as_tensor(expected_mean, device=device)
    student_std = torch.as_tensor(expected_std, device=device)
    teacher_mean = torch.as_tensor(pred_mean, device=device)
    teacher_std = torch.as_tensor(pred_std, device=device)
    with np.load(args.masks) as mask_file:
        gt_masks = {name: mask_file[name].copy() for name in ("gt_train", "gt_val", "gt_test")}
        pred_masks = {
            split: mask_file[f"pred_{split}"].copy() if f"pred_{split}" in mask_file else None
            for split in ("val", "test")
        }
    # Spread validation over the entire nue2 GT file, rather than taking the
    # first contiguous block of its event ordering.
    gt_val_small = gt_masks["gt_val"] & (np.arange(len(gt_masks["gt_val"])) % 6 == 0)

    def dataset(
        path,
        anchors,
        split,
        keep=None,
        single_muatm_only=False,
        pred_thinning_prob=0.0,
        train_aug=False,
    ):
        return AdaptedTrackDataset(
            str(path),
            split,
            args.batch_size,
            args.max_seq_len,
            False,
            None,
            False,
            anchor_path=str(anchors),
            keep=keep,
            expected_mean=expected_mean,
            expected_std=expected_std,
            single_muatm_only=single_muatm_only,
            pred_thinning_prob=pred_thinning_prob,
            pred_threshold_min=args.pred_threshold_min,
            pred_threshold_max=args.pred_threshold_max,
            span_crop_prob=args.span_crop_prob if train_aug else 0.0,
            span_crop_min_keep=args.span_crop_min_keep,
            span_crop_max_keep=args.span_crop_max_keep,
            score_feature=args.score_feature,
            score_jitter=args.score_jitter if split == "train" else 0.0,
            rotate_z_prob=args.rotate_z_prob if train_aug else 0.0,
            time_jitter_ns=args.time_jitter_ns if train_aug else 0.0,
            event_time_jitter_ns=args.event_time_jitter_ns if train_aug else 0.0,
            coord_jitter_m=args.coord_jitter_m if train_aug else 0.0,
            charge_log_jitter=args.charge_log_jitter if train_aug else 0.0,
        )

    def clean_nuatm_dataset(split):
        return CleanParticleAngleDataset(
            str(args.clean_nuatm_data),
            split,
            args.batch_size,
            args.max_seq_len,
            False,
            None,
            False,
            expected_mean=expected_mean,
            expected_std=expected_std,
            particle_code=1,
        )

    clean_paired_keep = None
    if args.clean_paired_anchors is not None:
        with h5py.File(args.clean_paired_anchors, "r") as anchors:
            clean_paired_keep = {
                split: np.asarray(anchors[f"{split}/valid"], dtype=bool)
                for split in ("train", "val", "test")
            }

    def clean_paired_dataset(split):
        return dataset(
            args.clean_paired_data,
            args.clean_paired_anchors,
            split,
            keep=clean_paired_keep[split],
        )

    if args.eval_only:
        split = args.eval_split
        selection = gt_val_small if split == "val" else gt_masks["gt_test"]
        pred_eval = make_loader(
            dataset(
                args.pred_data,
                args.pred_anchors,
                split,
                keep=pred_masks[split],
                single_muatm_only=args.single_muatm_only,
            ),
            args.workers,
            False,
            args.seed,
        )
        gt_eval = make_loader(
            dataset(args.gt_data, args.gt_anchors, split, selection),
            args.workers,
            False,
            args.seed,
        )
        pred_result, pred_rows = evaluate(model, pred_eval, device, return_arrays=True)
        gt_result, gt_rows = evaluate(model, gt_eval, device, return_arrays=True)
        result = {f"pred_{split}": pred_result, f"gt_{split}_clean": gt_result}
        if args.clean_nuatm_data is not None:
            clean_eval = make_loader(clean_nuatm_dataset(split), args.workers, False, args.seed)
            result[f"clean_nuatm_{split}"] = evaluate_angle_only(model, clean_eval, device)
        if clean_paired_keep is not None and args.clean_paired_data is not None:
            clean_eval = make_loader(clean_paired_dataset(split), args.workers, False, args.seed)
            result[f"clean_paired_{split}"] = evaluate(model, clean_eval, device)
        if args.single_muatm_only or pred_masks[split] is not None:
            pred_full = make_loader(
                dataset(args.pred_data, args.pred_anchors, split),
                args.workers,
                False,
                args.seed,
            )
            result[f"pred_{split}_full_diagnostic"] = evaluate(model, pred_full, device)
        (args.output_dir / "eval.json").write_text(json.dumps(result, indent=2) + "\n")
        if split == "test":
            np.savez_compressed(
                args.output_dir / "test_residuals.npz",
                pred_angle=pred_rows["angle"],
                pred_line=pred_rows["line"],
                gt_angle=gt_rows["angle"],
                gt_line=gt_rows["line"],
            )
        print(json.dumps(result, indent=2), flush=True)
        return

    pred_val = make_loader(
        dataset(
            args.pred_data,
            args.pred_anchors,
            "val",
            keep=pred_masks["val"],
            single_muatm_only=args.single_muatm_only,
        ),
        args.workers,
        False,
        args.seed,
    )
    gt_val = make_loader(
        dataset(args.gt_data, args.gt_anchors, "val", gt_val_small),
        args.workers,
        False,
        args.seed,
    )
    clean_nuatm_val = (
        make_loader(clean_nuatm_dataset("val"), args.workers, False, args.seed)
        if args.clean_nuatm_data is not None
        else None
    )
    clean_paired_val = (
        make_loader(clean_paired_dataset("val"), args.workers, False, args.seed)
        if clean_paired_keep is not None and args.clean_paired_data is not None
        else None
    )

    pred_train = infinite_batches(
        make_loader(
            dataset(
                args.pred_data,
                args.pred_anchors,
                "train",
                single_muatm_only=args.single_muatm_only,
                pred_thinning_prob=args.pred_thinning_prob,
                train_aug=True,
            ),
            args.workers,
            True,
            args.seed,
        )
    )
    gt_train = infinite_batches(
        make_loader(
            dataset(args.gt_data, args.gt_anchors, "train", gt_masks["gt_train"]),
            args.workers,
            True,
            args.seed + 1,
        )
    )
    clean_nuatm_train = (
        infinite_batches(
            make_loader(clean_nuatm_dataset("train"), args.workers, True, args.seed + 2)
        )
        if args.clean_nuatm_weight > 0
        else None
    )
    clean_paired_train = (
        infinite_batches(
            make_loader(clean_paired_dataset("train"), args.workers, True, args.seed + 3)
        )
        if args.clean_paired_weight > 0
        else None
    )
    if args.score_column_only:
        model.requires_grad_(False)
        score_weight = model.backbone.embed[0].weight
        score_weight.requires_grad_(True)
        gradient_mask = torch.zeros_like(score_weight)
        gradient_mask[:, -1] = 1
        score_weight.register_hook(lambda gradient: gradient * gradient_mask)
        optimizer = torch.optim.AdamW(
            [{"params": [score_weight], "lr": args.score_column_lr}],
            weight_decay=0.0,
            fused=device.type == "cuda",
        )
    elif args.graph_only:
        model.requires_grad_(False)
        model.backbone.graph_blocks.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            [{"params": model.backbone.graph_blocks.parameters(), "lr": args.graph_lr}],
            weight_decay=1e-4,
            fused=device.type == "cuda",
        )
    elif args.direction_head_only:
        model.backbone.requires_grad_(False)
        model.backbone.head.requires_grad_(True)
    if not args.score_column_only:
        if not args.graph_only:
            if args.graph_layers:
                backbone_params = [
                    value
                    for name, value in model.backbone.named_parameters()
                    if not name.startswith("graph_blocks.")
                ]
                groups = [
                    {"params": backbone_params, "lr": args.backbone_lr},
                    {
                        "params": model.backbone.graph_blocks.parameters(),
                        "lr": args.graph_lr,
                    },
                ]
            elif args.physics_features:
                first_weight = model.backbone.embed[0].weight
                original_features = first_weight.shape[1] - model.backbone.extra_features
                gradient_mask = torch.zeros_like(first_weight)
                gradient_mask[:, original_features:] = 1.0
                first_weight.register_hook(lambda gradient: gradient * gradient_mask)
                backbone_params = [
                    value
                    for name, value in model.backbone.named_parameters()
                    if name != "embed.0.weight"
                ]
                groups = [
                    {"params": backbone_params, "lr": args.backbone_lr},
                    {
                        "params": [first_weight],
                        "lr": args.physics_input_lr,
                        "weight_decay": 0.0,
                    },
                ]
            else:
                groups = [
                    {
                        "params": model.backbone.head.parameters()
                        if args.direction_head_only
                        else model.backbone.parameters(),
                        "lr": args.backbone_lr,
                    }
                ]
            groups.append({"params": model.point_head.parameters(), "lr": args.point_lr})
            optimizer = torch.optim.AdamW(
                groups,
                weight_decay=1e-4,
                fused=device.type == "cuda",
            )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    initial = {
        "pred_val": evaluate(model, pred_val, device),
        "gt_val_clean": evaluate(model, gt_val, device),
    }
    if clean_nuatm_val is not None:
        initial["clean_nuatm_val"] = evaluate_angle_only(model, clean_nuatm_val, device)
    if clean_paired_val is not None:
        initial["clean_paired_val"] = evaluate(model, clean_paired_val, device)
    initial["step"] = 0
    initial["score"] = validation_score(initial, args.pred_score_weight, args.muatm_score_weight)
    (args.output_dir / "val_step_0.json").write_text(json.dumps(initial, indent=2) + "\n")
    print("validation " + json.dumps(initial, sort_keys=True), flush=True)
    baseline = initial["pred_val"]["direction_deg"]["q68"]
    limit = args.max_pred_q68 if args.max_pred_q68 is not None else 1.05 * baseline
    initial_gt_q68 = initial["gt_val_clean"]["direction_deg"]["q68"]
    initial_admissible = baseline <= limit and (
        args.max_gt_q68 is None or initial_gt_q68 <= args.max_gt_q68
    )
    best_score = initial["score"] if initial_admissible else float("inf")
    for step in range(1, args.steps + 1):
        model.train()
        if args.score_column_only:
            model.eval()
        elif args.graph_only:
            model.eval()
            model.backbone.graph_blocks.train()
        elif args.direction_head_only:
            model.backbone.embed.eval()
            model.backbone.encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        losses = {}
        for name, source, weight in (
            ("pred", pred_train, 1 - args.gt_weight),
            ("gt", gt_train, args.gt_weight),
        ):
            x, target_dir, mask, particle, target_point, valid = next_nonempty(source)
            x = x.to(device, non_blocking=True)
            target_dir = target_dir.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            target_point = target_point.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                direction, point = model(x, mask)
                angular = loss_function(
                    direction,
                    target_dir,
                    args.pred_direction_loss if name == "pred" else "trimmed_angular",
                    args.pred_direction_power if name == "pred" else 1.0,
                    None,
                    args.pred_direction_cap_deg if name == "pred" else 8.0,
                )
                if name == "pred":
                    angular = angular * args.pred_direction_scale
                transverse = point_loss(
                    point,
                    target_point,
                    target_dir,
                    valid,
                    args.anchor_scale_m,
                    "line_huber",
                )
                loss = weight * (angular + args.point_weight * transverse)
                if name == "pred" and teacher is not None:
                    with torch.no_grad():
                        teacher_dir, teacher_point = teacher(x, mask)
                        if muatm_teacher is not None and torch.any(particle == 0):
                            teacher_x = (
                                x * student_std + student_mean - teacher_mean
                            ) / teacher_std
                            teacher_x = teacher_x * mask.unsqueeze(-1)
                            mu_dir, mu_point = muatm_teacher(teacher_x, mask)
                            use_mu = (particle.to(device) == 0).unsqueeze(-1)
                            teacher_dir = torch.where(use_mu, mu_dir, teacher_dir)
                            teacher_point = torch.where(use_mu, mu_point, teacher_point)
                    direction_distill = F.l1_loss(direction.float(), teacher_dir.float())
                    point_distill = F.smooth_l1_loss(
                        (point.float() - teacher_point.float()) / args.anchor_scale_m,
                        torch.zeros_like(point.float()),
                        beta=0.1,
                    )
                    loss = loss + weight * args.teacher_weight * (
                        direction_distill + args.point_weight * point_distill
                    )
            scaler.scale(loss).backward()
            losses[name] = float(loss.detach())
        if clean_nuatm_train is not None:
            x, target_dir, mask, _particle = next_nonempty(clean_nuatm_train)
            x = x.to(device, non_blocking=True)
            target_dir = target_dir.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                direction, _point = model(x, mask)
                clean_angular = loss_function(
                    direction, target_dir, "trimmed_angular", 1.0, None, 8.0
                )
                clean_loss = args.clean_nuatm_weight * clean_angular
            scaler.scale(clean_loss).backward()
            losses["clean_nuatm"] = float(clean_loss.detach())
        if clean_paired_train is not None:
            x, target_dir, mask, _particle, target_point, valid = next_nonempty(clean_paired_train)
            x = x.to(device, non_blocking=True)
            target_dir = target_dir.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            target_point = target_point.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                direction, point = model(x, mask)
                clean_angular = loss_function(
                    direction, target_dir, "trimmed_angular", 1.0, None, 8.0
                )
                clean_transverse = point_loss(
                    point,
                    target_point,
                    target_dir,
                    valid,
                    args.anchor_scale_m,
                    "line_huber",
                )
                clean_loss = args.clean_paired_weight * (
                    clean_angular + args.point_weight * clean_transverse
                )
            scaler.scale(clean_loss).backward()
            losses["clean_paired"] = float(clean_loss.detach())
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step % 50 == 0:
            print(f"step={step} losses={losses}", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            result = {
                "pred_val": evaluate(model, pred_val, device),
                "gt_val_clean": evaluate(model, gt_val, device),
            }
            if clean_nuatm_val is not None:
                result["clean_nuatm_val"] = evaluate_angle_only(model, clean_nuatm_val, device)
            if clean_paired_val is not None:
                result["clean_paired_val"] = evaluate(model, clean_paired_val, device)
            pred_q68 = result["pred_val"]["direction_deg"]["q68"]
            gt_q68 = result["gt_val_clean"]["direction_deg"]["q68"]
            # Primary: clean nue2 direction. Reject candidates that damage
            # all-particle direction by more than 5% on its own validation set.
            admissible = pred_q68 <= limit and (
                args.max_gt_q68 is None or gt_q68 <= args.max_gt_q68
            )
            score = validation_score(result, args.pred_score_weight, args.muatm_score_weight)
            result["step"] = step
            result["score"] = score
            result["admissible"] = admissible
            print("validation " + json.dumps(result, sort_keys=True), flush=True)
            (args.output_dir / f"val_step_{step}.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            if admissible and score < best_score:
                best_score = score
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": vars(args),
                        "metrics": result,
                        "step": step,
                    },
                    args.output_dir / "best.pt",
                )
    print(f"best_score={best_score}", flush=True)


if __name__ == "__main__":
    main()
