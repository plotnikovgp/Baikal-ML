#!/usr/bin/env python3
"""Time-oriented, azimuth-equivariant hit-pair correction to one v4 model.

Only a fixed maximum of 32 time-quantile hits are used, making inference
bounded on the CPU. The residual head starts at exactly zero; the trained
base network and point head are retained in the same single model.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PairVoteResidual(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        feature_mean: Tensor,
        feature_std: Tensor,
        *,
        sample_hits: int = 32,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        if sample_hits < 2:
            raise ValueError("At least two sampled hits are required")
        self.base = base
        self.base.requires_grad_(False)
        self.sample_hits = sample_hits
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean).float().view(1, 1, 5))
        self.register_buffer("feature_std", torch.as_tensor(feature_std).float().view(1, 1, 5))
        upper_i, upper_j = torch.triu_indices(sample_hits, sample_hits, offset=1)
        self.register_buffer("upper_i", upper_i)
        self.register_buffer("upper_j", upper_j)
        self.edge_head = nn.Sequential(
            nn.Linear(10, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )
        nn.init.zeros_(self.edge_head[-1].weight)
        nn.init.zeros_(self.edge_head[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def pair_vote(self, x: Tensor, mask: Tensor, base_direction: Tensor) -> Tensor:
        with torch.autocast("cuda", enabled=False):
            raw = x[..., :5].float() * self.feature_std + self.feature_mean
            times = raw[..., 1].masked_fill(~mask, float("inf"))
            order = torch.argsort(times, dim=1)
            counts = mask.sum(dim=1).clamp_min(1)
            used = counts.clamp_max(self.sample_hits)
            positions = torch.arange(self.sample_hits, device=x.device)[None, :]
            quantile = (
                (
                    positions.float()
                    * (counts - 1)[:, None].float()
                    / (used - 1).clamp_min(1)[:, None].float()
                )
                .round()
                .long()
                .clamp_max(mask.shape[1] - 1)
            )
            selected = torch.gather(order, 1, quantile)
            sampled = torch.gather(raw, 1, selected[..., None].expand(-1, -1, 5))
            sampled_valid = positions < used[:, None]
            a, b = sampled[:, self.upper_i], sampled[:, self.upper_j]
            delta = b[..., 2:5] - a[..., 2:5]
            dt = b[..., 1] - a[..., 1]
            distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-3)
            dxy = torch.linalg.vector_norm(delta[..., :2], dim=-1)
            unit = delta / distance[..., None]
            base_direction = base_direction.float()
            alignment = (unit * base_direction[:, None]).sum(dim=-1)
            cross_z = (
                base_direction[:, None, 0] * unit[..., 1]
                - base_direction[:, None, 1] * unit[..., 0]
            )
            edge = torch.stack(
                (
                    torch.log1p(dt.clamp_min(0)) / 6.0,
                    dxy / 100.0,
                    delta[..., 2] / 100.0,
                    distance / 100.0,
                    (dt - distance / 0.225) / 200.0,
                    torch.tanh(a[..., 0] / 20.0),
                    torch.tanh(b[..., 0] / 20.0),
                    torch.exp(-torch.square(dxy / 2.0)),
                    alignment,
                    cross_z,
                ),
                dim=-1,
            )
            logits_and_coeff = self.edge_head(edge)
            pair_valid = (
                sampled_valid[:, self.upper_i]
                & sampled_valid[:, self.upper_j]
                & (dt > 1e-3)
                & (distance > 1e-2)
            )
            logits = logits_and_coeff[..., 0].masked_fill(~pair_valid, -1e4)
            weights = torch.softmax(logits, dim=1) * pair_valid.float()
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            coefficients = torch.tanh(logits_and_coeff[..., 1:])
            rotated_xy = torch.stack((-unit[..., 1], unit[..., 0]), dim=-1)
            vote_xy = coefficients[..., 0:1] * unit[..., :2] + coefficients[..., 1:2] * rotated_xy
            vote_z = coefficients[..., 2:3] * unit[..., 2:3]
            vote = torch.cat((vote_xy, vote_z), dim=-1)
            return 0.5 * (weights[..., None] * vote).sum(dim=1)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        base_direction, point = self.base(x, mask)
        correction = self.pair_vote(x, mask, base_direction)
        direction = torch.nn.functional.normalize(base_direction.float() + correction, dim=1)
        return direction, point
