#!/usr/bin/env python3
"""One-model, z-rotation-equivariant photon-time residual correction for v4.

The frozen base predicts a line. A small per-hit head sees charge, physical
along/perpendicular distances, and Cherenkov arrival-time residual to that
line; scalar weights turn transverse hit vectors into direction/point votes.
The last layer is zero-initialized, exactly preserving v4 at step zero.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PhotonResidualUpgrade(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        feature_mean: Tensor,
        feature_std: Tensor,
        *,
        hidden: int = 64,
        refractive_index: float = 1.36,
        direction_scale: float = 0.30,
        point_scale_m: float = 20.0,
    ) -> None:
        super().__init__()
        if refractive_index <= 1 or direction_scale <= 0 or point_scale_m <= 0:
            raise ValueError("Invalid physical model or correction scale")
        self.base = base
        self.base.requires_grad_(False)
        self.refractive_index = refractive_index
        self.direction_scale = direction_scale
        self.point_scale_m = point_scale_m
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean).float().view(1, 1, 5))
        self.register_buffer("feature_std", torch.as_tensor(feature_std).float().view(1, 1, 5))
        self.hit_head = nn.Sequential(
            nn.Linear(7, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )
        nn.init.zeros_(self.hit_head[-1].weight)
        nn.init.zeros_(self.hit_head[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def residual_votes(
        self, x: Tensor, mask: Tensor, base_direction: Tensor, base_point: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        with torch.autocast("cuda", enabled=False):
            raw = x[..., :5].float() * self.feature_std + self.feature_mean
            u = base_direction.float()
            p = base_point.float()
            weights = mask.float()
            counts = weights.sum(dim=1).clamp_min(1.0)
            positions = raw[..., 2:5]
            centroid = (positions * weights[..., None]).sum(dim=1) / counts[:, None]
            delta = positions - centroid[:, None] - p[:, None]
            along = (delta * u[:, None]).sum(dim=-1)
            transverse = delta - along[..., None] * u[:, None]
            distance = torch.linalg.vector_norm(transverse, dim=-1).clamp_min(0.1)
            expected_ns = (along + (self.refractive_index**2 - 1.0) ** 0.5 * distance) / 0.299792458
            residual = raw[..., 1] - expected_ns
            centered_time = residual.masked_fill(~mask, float("nan")).nanmedian(dim=1).values
            centered = (residual - centered_time[:, None]).masked_fill(~mask, 0.0)
            mad = centered.abs().masked_fill(~mask, float("nan")).nanmedian(dim=1).values
            features = torch.stack(
                (
                    (along / 100.0).clamp(-5, 5),
                    (distance / 100.0).clamp(0, 5),
                    (centered / 50.0).clamp(-10, 10),
                    torch.log1p(raw[..., 0].clamp_min(0)) / 5.0,
                    (mad[:, None] / 50.0).expand_as(along).clamp(0, 10),
                    (counts[:, None] / 30.0).expand_as(along).clamp(0, 10),
                    u[:, None, 2].expand_as(along),
                ),
                dim=-1,
            )
            coefficients = torch.tanh(self.hit_head(features).float())
            basis = transverse / distance[..., None]
            longitudinal = (along / 100.0).clamp(-5, 5)
            direction_weight = coefficients[..., 0] + coefficients[..., 1] * longitudinal
            point_weight = coefficients[..., 2] + coefficients[..., 3] * longitudinal
            direction_vote = (direction_weight[..., None] * basis * weights[..., None]).sum(
                dim=1
            ) / counts[:, None]
            point_vote = (point_weight[..., None] * basis * weights[..., None]).sum(dim=1) / counts[
                :, None
            ]
            return direction_vote, point_vote, mad

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        base_direction, base_point = self.base(x, mask)
        direction_vote, point_vote, _mad = self.residual_votes(x, mask, base_direction, base_point)
        direction = torch.nn.functional.normalize(
            base_direction.float() + self.direction_scale * direction_vote, dim=1
        )
        point = base_point.float() + self.point_scale_m * point_vote
        point = point - direction * (point * direction).sum(dim=1, keepdim=True)
        return direction, point
