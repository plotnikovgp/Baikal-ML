#!/usr/bin/env python3
"""Training-only vertical-span hit crop with an invariant truth-line target."""

from __future__ import annotations

import torch
from torch import Tensor


def crop_vertical_span(
    x: Tensor,
    mask: Tensor,
    direction: Tensor,
    point: Tensor,
    valid: Tensor,
    mean: Tensor,
    std: Tensor,
    *,
    probability: float,
    min_keep_fraction: float,
    max_keep_fraction: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Randomly trim extreme-z hits, preserving >=8 hits, >=2 strings, and line.

    The input and point target are relative to the same current hit centroid.
    Rebase the target after a valid crop; this leaves the physical truth line
    unchanged. Padded rows stay masked and zeroed. Spatial coordinates are
    converted to metres before the cut or centroid calculation.
    """
    if not 0 <= probability <= 1:
        raise ValueError("Crop probability must be in [0, 1]")
    if not 0 < min_keep_fraction <= max_keep_fraction <= 1:
        raise ValueError("Invalid kept-hit fraction interval")
    if not len(x) or probability == 0:
        return x, mask, point
    if x.ndim != 3 or x.shape[-1] < 5 or mask.shape != x.shape[:2]:
        raise ValueError("Expected batched hit features and a matching mask")
    positions = x[..., 2:5] * std[2:5] + mean[2:5]
    counts = mask.sum(dim=1)
    order = torch.argsort(positions[..., 2].masked_fill(~mask, torch.inf), dim=1)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(mask.shape[1], device=x.device)[None].expand_as(order))
    fraction = torch.empty(len(x), device=x.device).uniform_(min_keep_fraction, max_keep_fraction)
    remove = torch.round(counts.float() * (1 - fraction)).long()
    lower = torch.floor(torch.rand(len(x), device=x.device) * (remove + 1).float()).long()
    upper = remove - lower
    candidate = mask & (ranks >= lower[:, None]) & (ranks < (counts - upper)[:, None])
    enough_hits = candidate.sum(dim=1) >= 8
    first_index = candidate.int().argmax(dim=1)
    first_xy = positions[torch.arange(len(x), device=x.device), first_index, :2]
    distinct_string = (
        (torch.linalg.vector_norm(positions[..., :2] - first_xy[:, None], dim=-1) > 1.0) & candidate
    ).any(dim=1)
    apply = (
        (torch.rand(len(x), device=x.device) < probability)
        & (remove > 0)
        & enough_hits
        & distinct_string
    )
    new_mask = torch.where(apply[:, None], candidate, mask)
    old_count = counts.clamp_min(1)[:, None]
    new_count = new_mask.sum(dim=1).clamp_min(1)[:, None]
    old_centroid = (positions * mask[..., None]).sum(dim=1) / old_count
    new_centroid = (positions * new_mask[..., None]).sum(dim=1) / new_count
    rebased = point + old_centroid - new_centroid
    rebased -= direction * (rebased * direction).sum(dim=1, keepdim=True)
    new_point = torch.where((apply & valid)[:, None], rebased, point)
    return x * new_mask[..., None], new_mask, new_point
