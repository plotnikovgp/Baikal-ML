#!/usr/bin/env python3
"""One CPU-loadable model with frozen photon correction and uncertainty head.

The existing uncertainty head still sees the original v4 features and emits
its original scale predictions.  Direction and point are corrected by the
frozen physical-time vote head.  Validation calibration must be recomputed
for those corrected errors before emitting coverage radii.
"""

from __future__ import annotations

from pathlib import Path

import torch
from bundle_universal_model import build_model as build_v4_model
from photon_residual_upgrade import PhotonResidualUpgrade
from torch import Tensor, nn
from torch.nn import functional as F


class PhotonCorrectedUncertainty(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.base_model = build_v4_model(config["base_model_config"])
        photon = config["photon"]
        # Only residual_votes is called. Identity avoids storing a second copy
        # of the same backbone in the single-file bundle.
        self.photon_votes = PhotonResidualUpgrade(
            nn.Identity(),
            torch.tensor(photon["feature_mean"], dtype=torch.float32),
            torch.tensor(photon["feature_std"], dtype=torch.float32),
            hidden=photon["hidden"],
            refractive_index=photon["refractive_index"],
            direction_scale=photon["direction_scale"],
            point_scale_m=photon["point_scale_m"],
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.base.eval()
        self.photon_votes.eval()
        return self

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        base_direction, base_point, log_angle, log_line = self.base_model(x, mask)
        direction_vote, point_vote, _ = self.photon_votes.residual_votes(
            x, mask, base_direction, base_point
        )
        direction = F.normalize(
            base_direction.float() + self.photon_votes.direction_scale * direction_vote,
            dim=1,
        )
        point = base_point.float() + self.photon_votes.point_scale_m * point_vote
        point = point - direction * (point * direction).sum(dim=1, keepdim=True)
        return direction, point, log_angle, log_line


def build_model(config: dict) -> PhotonCorrectedUncertainty:
    return PhotonCorrectedUncertainty(config)


def load_frozen_sources(
    bundle_path: Path, photon_checkpoint_path: Path
) -> tuple[PhotonCorrectedUncertainty, dict, dict]:
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    if bundle["format_version"] != 1:
        raise ValueError("Expected the original v4 all-head bundle (format 1)")
    # This trusted local training checkpoint contains pathlib objects in args.
    photon_checkpoint = torch.load(photon_checkpoint_path, map_location="cpu", weights_only=False)
    original_state = bundle["state_dict"]
    photon_state = photon_checkpoint["model"]
    original_base = {k: v for k, v in original_state.items() if k.startswith("base.")}
    photon_base = {k: v for k, v in photon_state.items() if k.startswith("base.")}
    if set(original_base) != set(photon_base) or any(
        not torch.equal(original_base[key], photon_base[key]) for key in original_base
    ):
        raise ValueError("Photon checkpoint is not on the exact v4 backbone")
    prep = bundle["preprocessing"]
    mean = torch.tensor(prep["input_norm_mean"], dtype=torch.float32)
    std = torch.tensor(prep["input_norm_std"], dtype=torch.float32)
    if not torch.equal(photon_state["feature_mean"].flatten(), mean) or not torch.equal(
        photon_state["feature_std"].flatten(), std
    ):
        raise ValueError("Photon checkpoint normalization differs from v4")
    config = {
        "base_model_config": bundle["model_config"],
        "photon": {
            "feature_mean": prep["input_norm_mean"],
            "feature_std": prep["input_norm_std"],
            "hidden": int(photon_checkpoint["args"]["hidden"]),
            "refractive_index": 1.36,
            "direction_scale": 0.30,
            "point_scale_m": 20.0,
        },
    }
    model = build_model(config)
    model.base_model.load_state_dict(original_state, strict=True)
    hit_head = {
        key.removeprefix("hit_head."): value
        for key, value in photon_state.items()
        if key.startswith("hit_head.")
    }
    model.photon_votes.hit_head.load_state_dict(hit_head, strict=True)
    model.base_model.requires_grad_(False)
    model.photon_votes.requires_grad_(False)
    model.eval()
    return model, bundle, config
