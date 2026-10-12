#!/usr/bin/env python3
"""One-pass Transformer with explicit space-time track features.

The extra input columns are zero-initialized when upgrading the existing
Transformer, so the starting checkpoint has exactly the same predictions.
PCA is an input feature only; it never replaces the learned direction.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from train_angle_signal import SetTransformerDirection, engineered_features, masked_mean


class PhysicsFeatureTransformerDirection(SetTransformerDirection):
    extra_features = 6

    def __init__(
        self,
        hidden: int,
        layers: int,
        heads: int,
        ff_mult: int,
        dropout: float,
        feature_mode: str,
    ) -> None:
        super().__init__(hidden, layers, heads, ff_mult, dropout, feature_mode)
        original = self.embed[0]
        self.embed[0] = nn.Linear(original.in_features + self.extra_features, hidden)
        self.register_buffer("feature_mean", torch.zeros(1, 1, 5))
        self.register_buffer("feature_std", torch.ones(1, 1, 5))

    def set_feature_norm(self, mean: Tensor, std: Tensor) -> None:
        self.feature_mean.copy_(
            torch.as_tensor(mean, device=self.feature_mean.device).view(1, 1, 5)
        )
        self.feature_std.copy_(torch.as_tensor(std, device=self.feature_std.device).view(1, 1, 5))

    def physical_features(self, x: Tensor, mask: Tensor) -> Tensor:
        # These are physical metres and nanoseconds before normalization.
        # Keep small 3x3 eigensystems in float32, including under AMP.
        with torch.autocast("cuda", enabled=False):
            raw = x[..., :5].float() * self.feature_std + self.feature_mean
            weights = mask.float()
            counts = weights.sum(dim=1).clamp_min(1.0)
            position = raw[..., 2:5]
            times = raw[..., 1]
            position_center = (position * weights[..., None]).sum(dim=1) / counts[:, None]
            time_center = (times * weights).sum(dim=1) / counts
            centered_position = (position - position_center[:, None]) * weights[..., None]
            centered_time = (times - time_center[:, None]) * weights
            spatial_covariance = (
                torch.einsum("bni,bnj->bij", centered_position, centered_position)
                / counts[:, None, None]
            )
            time_covariance = (
                torch.einsum("bni,bn->bi", centered_position, centered_time) / counts[:, None]
            )
            _eigenvalues, eigenvectors = torch.linalg.eigh(spatial_covariance)
            axis = eigenvectors[..., -1]
            sign = torch.where(
                (axis * time_covariance).sum(dim=1) < 0,
                -torch.ones_like(axis[:, 0]),
                torch.ones_like(axis[:, 0]),
            )
            axis = axis * sign[:, None]
            longitudinal = (centered_position * axis[:, None]).sum(dim=-1)
            transverse = torch.linalg.vector_norm(
                centered_position - longitudinal[..., None] * axis[:, None], dim=-1
            )
            # A crude propagation reference, not a claimed Cherenkov fit.
            time_residual = centered_time - longitudinal / 0.299792458
            extra = torch.cat(
                (
                    (longitudinal / 100.0)[..., None],
                    (transverse / 100.0)[..., None],
                    (time_residual / 100.0).clamp(-10.0, 10.0)[..., None],
                    axis[:, None].expand(-1, x.shape[1], -1),
                ),
                dim=-1,
            )
        return extra * mask[..., None]

    def forward(self, x: Tensor, mask: Tensor, return_features: bool = False):
        base = engineered_features(x, mask) if self.feature_mode == "engineered" else x
        features = torch.cat((base, self.physical_features(x, mask)), dim=-1)
        h = self.embed(features)
        cls = self.cls.expand(len(h), -1, -1)
        h = torch.cat((cls, h), dim=1)
        full_mask = torch.cat(
            (torch.ones((len(mask), 1), dtype=torch.bool, device=mask.device), mask),
            dim=1,
        )
        h = self.encoder(h, src_key_padding_mask=~full_mask)
        pooled = torch.cat((h[:, 0], masked_mean(h[:, 1:], mask)), dim=1)
        direction = torch.nn.functional.normalize(self.head(pooled), dim=1)
        return (direction, pooled) if return_features else direction


def upgrade_track_model_physics_features(model: nn.Module) -> nn.Module:
    old = model.backbone
    if not isinstance(old, SetTransformerDirection) or isinstance(
        old, PhysicsFeatureTransformerDirection
    ):
        raise TypeError("Expected an unmodified SetTransformerDirection backbone")
    hidden = old.embed[0].out_features
    first_layer = old.encoder.layers[0]
    new = PhysicsFeatureTransformerDirection(
        hidden,
        len(old.encoder.layers),
        first_layer.self_attn.num_heads,
        first_layer.linear1.out_features // hidden,
        first_layer.dropout.p,
        old.feature_mode,
    ).to(next(old.parameters()).device)
    previous = old.state_dict()
    upgraded = new.state_dict()
    for key, value in previous.items():
        if key == "embed.0.weight":
            upgraded[key].zero_()
            upgraded[key][:, : value.shape[1]].copy_(value)
        else:
            upgraded[key].copy_(value)
    new.load_state_dict(upgraded, strict=True)
    model.backbone = new
    return model
