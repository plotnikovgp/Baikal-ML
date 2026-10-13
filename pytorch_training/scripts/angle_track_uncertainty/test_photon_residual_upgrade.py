"""Invariants and trainability of the physical time-residual correction."""

import torch
from photon_residual_upgrade import PhotonResidualUpgrade
from torch import nn


class FixedLine(nn.Module):
    def __init__(self, direction: torch.Tensor, point: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("direction", direction)
        self.register_buffer("point", point)

    def forward(self, x, mask):
        del mask
        return self.direction[None].expand(len(x), -1), self.point[None].expand(len(x), -1)


def event() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(15)
    x = torch.randn(2, 12, 5)
    x[..., 0] = torch.rand(2, 12) * 10 + 1
    x[..., 1] = torch.linspace(-100, 100, 12)[None] + torch.randn(2, 12) * 4
    x[..., 2:5] *= 25
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[0, -2:] = False
    x[0, -2:] = 0
    return x, mask


def model() -> PhotonResidualUpgrade:
    base = FixedLine(
        torch.nn.functional.normalize(torch.tensor([0.3, -0.2, 0.9]), dim=0),
        torch.tensor([4.0, 2.0, -1.0]),
    )
    return PhotonResidualUpgrade(base, torch.zeros(5), torch.ones(5), hidden=16)


def test_identity_initialization_and_gradients():
    x, mask = event()
    candidate = model()
    direction, point = candidate(x, mask)
    expected_direction, expected_point = candidate.base(x, mask)
    expected_point = expected_point - expected_direction * (
        expected_point * expected_direction
    ).sum(dim=1, keepdim=True)
    torch.testing.assert_close(direction, expected_direction, atol=1e-7, rtol=0)
    torch.testing.assert_close(point, expected_point, atol=1e-6, rtol=0)
    (direction[:, 0].sum() + point[:, 0].sum()).backward()
    assert candidate.hit_head[-1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in candidate.base.parameters())


def test_time_translation_and_z_rotation_equivariance():
    x, mask = event()
    candidate = model()
    with torch.no_grad():
        candidate.hit_head[-1].weight.normal_(0, 0.1)
        candidate.hit_head[-1].bias.normal_(0, 0.1)
    direction, point = candidate(x, mask)
    shifted = x.clone()
    shifted[..., 1] += 1000
    shifted[~mask] = 0
    shifted_direction, shifted_point = candidate(shifted, mask)
    torch.testing.assert_close(shifted_direction, direction, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(shifted_point, point, atol=1e-4, rtol=1e-5)

    angle = torch.tensor(0.7)
    c, s = torch.cos(angle), torch.sin(angle)
    rotation = torch.stack(
        (
            torch.stack((c, -s, torch.zeros_like(c))),
            torch.stack((s, c, torch.zeros_like(c))),
            torch.tensor([0.0, 0.0, 1.0]),
        )
    )
    rotated = x.clone()
    rotated[..., 2:5] = x[..., 2:5] @ rotation.T
    rotated_base = FixedLine(
        candidate.base.direction @ rotation.T, candidate.base.point @ rotation.T
    )
    rotated_model = PhotonResidualUpgrade(rotated_base, torch.zeros(5), torch.ones(5), hidden=16)
    rotated_model.hit_head.load_state_dict(candidate.hit_head.state_dict())
    rotated_direction, rotated_point = rotated_model(rotated, mask)
    torch.testing.assert_close(rotated_direction, direction @ rotation.T, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(rotated_point, point @ rotation.T, atol=2e-4, rtol=1e-5)
