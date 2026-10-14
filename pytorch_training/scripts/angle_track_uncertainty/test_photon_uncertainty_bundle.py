"""The single-file wrapper preserves the existing correction and uncertainty."""

from __future__ import annotations

import torch
from photon_residual_upgrade import PhotonResidualUpgrade
from photon_uncertainty_bundle import build_model


def test_photon_wrapper_matches_standalone_correction() -> None:
    torch.manual_seed(19)
    config = {
        "base_model_config": {
            "hidden": 32,
            "layers": 1,
            "heads": 4,
            "ff_mult": 2,
            "dropout": 0.0,
            "feature_mode": "engineered",
            "anchor_scale_m": 50.0,
            "uncertainty_dropout": 0.0,
        },
        "photon": {
            "feature_mean": [0.0] * 5,
            "feature_std": [1.0] * 5,
            "hidden": 16,
            "refractive_index": 1.36,
            "direction_scale": 0.30,
            "point_scale_m": 20.0,
        },
    }
    combined = build_model(config).eval()
    standalone = PhotonResidualUpgrade(
        combined.base_model.base,
        torch.zeros(5),
        torch.ones(5),
        hidden=16,
    ).eval()
    standalone.hit_head.load_state_dict(combined.photon_votes.hit_head.state_dict())
    x = torch.randn(4, 12, 5)
    x[:, :, 0] = x[:, :, 0].abs() + 1
    x[:, :, 2:5] *= 30
    mask = torch.arange(12)[None] < torch.tensor([[12], [10], [9], [8]])
    x = x * mask.unsqueeze(-1)
    with torch.inference_mode():
        direction, point, log_angle, log_line = combined(x, mask)
        reference_direction, reference_point = standalone(x, mask)
        _, _, original_log_angle, original_log_line = combined.base_model(x, mask)
    torch.testing.assert_close(direction, reference_direction, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(point, reference_point, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(log_angle, original_log_angle, atol=0, rtol=0)
    torch.testing.assert_close(log_line, original_log_line, atol=0, rtol=0)
    assert torch.isfinite(direction).all()
    assert torch.isfinite(point).all()
