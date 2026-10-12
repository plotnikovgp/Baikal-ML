#!/usr/bin/env python3
"""Training-only physical perturbations with consistent direction/line labels."""

from __future__ import annotations

import math

import torch


def augment_track_batch(
    x: torch.Tensor,
    direction: torch.Tensor,
    mask: torch.Tensor,
    point: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    rotate_z_prob: float = 0.0,
    time_jitter_ns: float = 0.0,
    event_time_jitter_ns: float = 0.0,
    coord_jitter_m: float = 0.0,
    charge_log_jitter: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perturb selected hits; keep the physical line consistent with its centroid.

    ``point`` is the point on the truth line relative to the centroid of the
    selected hits. Spatial jitter changes that centroid, so we rebase point
    before any rotation. Rotation is around the perturbed event centroid.
    The optional sixth signal-score feature is passed through untouched.
    """
    if not len(x) or not any(
        (rotate_z_prob, time_jitter_ns, event_time_jitter_ns, coord_jitter_m, charge_log_jitter)
    ):
        return x, direction, point
    raw = x[..., :5] * std + mean
    weights = mask.unsqueeze(-1).to(raw.dtype)
    counts = weights.sum(dim=1).clamp_min(1)
    old_centroid = (raw[..., 2:5] * weights).sum(dim=1) / counts

    if event_time_jitter_ns:
        raw[..., 1] += torch.randn((len(raw), 1), dtype=raw.dtype) * event_time_jitter_ns
    if time_jitter_ns:
        raw[..., 1] += torch.randn_like(raw[..., 1]) * time_jitter_ns
    if coord_jitter_m:
        raw[..., 2:5] += torch.randn_like(raw[..., 2:5]) * coord_jitter_m
    if charge_log_jitter:
        scale = torch.exp(torch.randn((len(raw), 1), dtype=raw.dtype) * charge_log_jitter)
        raw[..., 0] *= scale

    centroid = (raw[..., 2:5] * weights).sum(dim=1) / counts
    point = point + old_centroid - centroid
    point = point - direction * (point * direction).sum(dim=1, keepdim=True)

    if rotate_z_prob:
        apply = torch.rand(len(raw)) < rotate_z_prob
        angle = torch.rand(len(raw), dtype=raw.dtype) * (2 * math.pi)
        angle = torch.where(apply, angle, torch.zeros_like(angle))
        cosine, sine = torch.cos(angle), torch.sin(angle)

        def rotate_xy(values: torch.Tensor) -> torch.Tensor:
            c = cosine[:, None] if values.ndim == 3 else cosine
            s = sine[:, None] if values.ndim == 3 else sine
            return torch.stack(
                (c * values[..., 0] - s * values[..., 1], s * values[..., 0] + c * values[..., 1]),
                dim=-1,
            )

        centered_xy = raw[..., 2:4] - centroid[:, None, :2]
        raw[..., 2:4] = rotate_xy(centered_xy)
        raw[..., 2:4] += centroid[:, None, :2]
        direction = torch.cat((rotate_xy(direction), direction[:, 2:3]), dim=1)
        point = torch.cat((rotate_xy(point), point[:, 2:3]), dim=1)

    out = ((raw - mean) / std) * weights
    if x.shape[-1] > 5:
        out = torch.cat((out, x[..., 5:]), dim=-1)
    return out, direction, point
