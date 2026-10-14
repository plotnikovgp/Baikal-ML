"""Training-only hit subsampling that preserves the target physical line."""

from __future__ import annotations

import torch


def crop_random_hits(
    x: torch.Tensor,
    mask: torch.Tensor,
    direction: torch.Tensor,
    point: torch.Tensor,
    valid: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    probability: float,
    min_keep: int = 8,
    max_keep: int = 12,
    attempts: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep 8-12 random hits, two strings, and rebase the point target.

    ``point`` is relative to the centroid of the currently retained hits.
    This can run after another hit-selection augmentation: no anchor-file
    centroid from the original event is assumed.
    """
    if not 0 <= probability <= 1 or min_keep < 2 or max_keep < min_keep or attempts < 1:
        raise ValueError("Invalid random-hit crop settings")
    if not len(x) or probability == 0:
        return x, mask, point
    counts = mask.sum(dim=1)
    active = (torch.rand(len(x), device=x.device) < probability) & (counts > min_keep)
    if not active.any():
        return x, mask, point

    positions_m = x[..., 2:5] * std[2:5] + mean[2:5]
    keep_upper = torch.minimum(counts - 1, torch.full_like(counts, max_keep))
    target = (
        min_keep
        + torch.floor(
            torch.rand(len(x), device=x.device) * (keep_upper - min_keep + 1).clamp_min(1)
        ).long()
    )
    target = torch.where(active, target, counts)
    remaining = active.clone()
    selected = mask.clone()
    for _ in range(attempts):
        if not remaining.any():
            break
        priority = torch.rand(mask.shape, device=x.device).masked_fill(~mask, 2.0)
        rank = priority.argsort(dim=1).argsort(dim=1)
        candidate = mask & (rank < target[:, None])
        first = candidate.long().argmax(dim=1)
        first_xy = positions_m[torch.arange(len(x), device=x.device), first, :2]
        has_second_string = (
            candidate
            & (torch.linalg.vector_norm(positions_m[..., :2] - first_xy[:, None], dim=-1) > 1.0)
        ).any(dim=1)
        accepted = remaining & has_second_string
        selected = torch.where(accepted[:, None], candidate, selected)
        remaining &= ~accepted

    changed = (selected != mask).any(dim=1)
    if not changed.any():
        return x, mask, point
    old_center = (positions_m * mask[..., None]).sum(dim=1) / counts.clamp_min(1)[:, None]
    new_center = (positions_m * selected[..., None]).sum(dim=1) / selected.sum(dim=1).clamp_min(1)[
        :, None
    ]
    rebased = point + old_center - new_center
    rebased -= direction * (rebased * direction).sum(dim=1, keepdim=True)
    point = torch.where((changed & valid)[:, None], rebased, point)
    return x * selected[..., None], selected, point
