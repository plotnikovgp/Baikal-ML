#!/usr/bin/env python3
"""Numerical invariants for geometry/time augmentation of track labels."""

import torch
from track_augmentation import augment_track_batch


def test_event_time_shift_uses_nanoseconds_after_denormalization() -> None:
    torch.manual_seed(5)
    x = torch.tensor(
        [[[0.5, 0.0, 0.0, 0.0, 0.0], [0.4, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]]
    )
    mask = torch.tensor([[True, True, False]])
    direction = torch.tensor([[0.0, 0.0, 1.0]])
    point = torch.tensor([[2.0, 3.0, 0.0]])
    mean = torch.tensor([0.0, 1000.0, 10.0, 20.0, 30.0])
    std = torch.tensor([1.0, 100.0, 10.0, 10.0, 10.0])
    same = augment_track_batch(x, direction, mask, point, mean, std)
    for before, after in zip((x, direction, point), same):
        torch.testing.assert_close(before, after)
    out, new_direction, new_point = augment_track_batch(
        x,
        direction,
        mask,
        point,
        mean,
        std,
        event_time_jitter_ns=5.0,
    )
    shift_ns = (out[0, :2, 1] - x[0, :2, 1]) * std[1]
    torch.testing.assert_close(shift_ns[0], shift_ns[1], atol=2e-4, rtol=0)
    assert 0 < abs(float(shift_ns[0])) < 20
    assert torch.all(out[0, 2] == 0)
    torch.testing.assert_close(new_direction, direction)
    torch.testing.assert_close(new_point, point)


def test_coordinate_jitter_rebases_the_same_physical_line() -> None:
    torch.manual_seed(17)
    x = torch.zeros(2, 5, 6)
    x[:, :, 2:5] = torch.randn(2, 5, 3) * 40
    x[:, :, 5] = 0.9
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    x *= mask.unsqueeze(-1)
    direction = torch.tensor([[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]])
    point = torch.tensor([[3.0, -2.0, 0.0], [0.0, 5.0, 0.0]])
    old_center = (x[..., 2:5] * mask.unsqueeze(-1)).sum(1) / mask.sum(1)[:, None]
    old_world = old_center + point
    out, new_direction, new_point = augment_track_batch(
        x,
        direction,
        mask,
        point,
        torch.zeros(5),
        torch.ones(5),
        coord_jitter_m=0.2,
        time_jitter_ns=1.0,
    )
    new_center = (out[..., 2:5] * mask.unsqueeze(-1)).sum(1) / mask.sum(1)[:, None]
    new_world = new_center + new_point
    transverse_change = new_world - old_world
    transverse_change -= direction * (transverse_change * direction).sum(1, keepdim=True)
    torch.testing.assert_close(
        transverse_change, torch.zeros_like(transverse_change), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(new_direction, direction)
    torch.testing.assert_close(out[..., 5], x[..., 5])
    assert torch.all(out[~mask] == 0)


def test_rotation_preserves_internal_geometry_and_track_distance() -> None:
    torch.manual_seed(31)
    x = torch.zeros(3, 4, 5)
    x[..., 2:5] = torch.randn(3, 4, 3) * 100
    mask = torch.ones(3, 4, dtype=torch.bool)
    direction = torch.tensor([[1.0, 0.0, 0.0]]).repeat(3, 1)
    point = torch.tensor([[0.0, 3.0, 4.0]]).repeat(3, 1)
    out, new_direction, new_point = augment_track_batch(
        x,
        direction,
        mask,
        point,
        torch.zeros(5),
        torch.ones(5),
        rotate_z_prob=1.0,
    )
    before_delta = x[:, 0, 2:5] - x[:, 1, 2:5]
    after_delta = out[:, 0, 2:5] - out[:, 1, 2:5]
    torch.testing.assert_close(
        before_delta.norm(dim=1), after_delta.norm(dim=1), atol=1e-5, rtol=1e-6
    )
    torch.testing.assert_close(new_direction.norm(dim=1), torch.ones(3), atol=1e-6, rtol=0)
    torch.testing.assert_close(new_point.norm(dim=1), point.norm(dim=1), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        (new_point * new_direction).sum(1), torch.zeros(3), atol=1e-6, rtol=0
    )
