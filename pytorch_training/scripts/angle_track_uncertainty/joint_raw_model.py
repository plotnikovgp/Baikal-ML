#!/usr/bin/env python3
"""One-pass signal selection, direction, and line-point reconstruction.

The hit encoder and classifier are warm-started from the existing signal/noise
model. Direction and line heads pool its contextual hit representations using
soft signal gates. This is one shared network at inference, not an ensemble.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor, nn


class JointRawReconstructor(nn.Module):
    def __init__(
        self,
        hidden: int = 128,
        layers: int = 5,
        ff_size: int = 512,
        legacy_float_mask: bool = True,
    ) -> None:
        super().__init__()
        self.legacy_float_mask = legacy_float_mask
        self.first_layer = nn.Linear(5, hidden)
        block = nn.TransformerEncoderLayer(hidden, 1, ff_size, dropout=0.0, batch_first=True)
        self.shared = nn.TransformerEncoder(block, layers)
        self.cls_head = nn.Linear(hidden, 2, bias=False)
        self.direction_head = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.point_head = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.point_head[-1].weight)
        nn.init.zeros_(self.point_head[-1].bias)
        self.register_buffer("feature_mean", torch.zeros(1, 1, 5))
        self.register_buffer("feature_std", torch.ones(1, 1, 5))

    def set_feature_norm(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.feature_mean.copy_(
            torch.as_tensor(mean, device=self.feature_mean.device).view(1, 1, 5)
        )
        self.feature_std.copy_(torch.as_tensor(std, device=self.feature_std.device).view(1, 1, 5))

    def load_signal_checkpoint(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "model" in state:
            state = state["model"]
        expected = {
            name
            for name in self.state_dict()
            if name.startswith(("first_layer.", "shared.", "cls_head."))
        }
        missing = expected - set(state)
        if missing:
            raise ValueError(f"Signal checkpoint lacks shared weights: {sorted(missing)[:8]}")
        new_state = self.state_dict()
        for name in expected:
            if new_state[name].shape != state[name].shape:
                raise ValueError(f"Shape mismatch for {name}")
            new_state[name].copy_(state[name])
        self.load_state_dict(new_state, strict=True)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # The source classifier was trained with a float 0/1 key mask. In
        # PyTorch this is additive rather than a hard boolean pad mask. Keep
        # that behavior by default for an exact warm start; also support a
        # corrected hard-mask ablation.
        key_pad = (~mask).float() if self.legacy_float_mask else ~mask
        hidden = self.shared(self.first_layer(x), src_key_padding_mask=key_pad)
        # Keep probabilities and geometric heads in FP32 under BF16 AMP;
        # angular errors near a few degrees are sensitive to BF16 rounding.
        with torch.autocast("cuda", enabled=False):
            hidden = hidden.float()
            hit_logit = self.cls_head(hidden)[..., 1]
            probability = torch.sigmoid(hit_logit)
            # Differentiable approximation to the current p>=0.70 hit cut.
            gate = torch.sigmoid((probability - 0.70) * 15.0) * mask.float()
            denominator = gate.sum(dim=1).clamp_min(1e-6)
            pooled = (hidden * gate[..., None]).sum(dim=1) / denominator[:, None]
            variance = (gate[..., None] * (hidden - pooled[:, None]).square()).sum(dim=1)
            variance = torch.sqrt(variance / denominator[:, None] + 1e-6)
            event_features = torch.cat((pooled, variance), dim=1)
            direction = torch.nn.functional.normalize(self.direction_head(event_features), dim=1)
            raw_offset = self.point_head(event_features) * 50.0
            detached_direction = direction.detach()
            point_offset = raw_offset - detached_direction * (raw_offset * detached_direction).sum(
                dim=1, keepdim=True
            )
            raw_position = (
                x[..., 2:5].float() * self.feature_std[..., 2:5] + self.feature_mean[..., 2:5]
            )
            centroid_m = (raw_position * gate[..., None]).sum(dim=1) / denominator[:, None]
        return direction, point_offset, centroid_m, hit_logit, probability


def build_pretrained_joint_model(
    checkpoint: Path, model_norm_source: Path, *, legacy_float_mask: bool = True
) -> JointRawReconstructor:
    model = JointRawReconstructor(legacy_float_mask=legacy_float_mask)
    model.load_signal_checkpoint(checkpoint)
    with h5py.File(model_norm_source, "r") as h5:
        mean = np.asarray(h5["norm_param/mean"], dtype=np.float32)
        std = np.asarray(h5["norm_param/std"], dtype=np.float32)
    model.set_feature_norm(mean, std)
    return model
