"""The physical-input upgrade preserves v4 predictions at initialization."""

import torch
from physics_feature_upgrade import upgrade_track_model_physics_features
from train_angle_signal import SetTransformerDirection
from train_track_anchor_v3 import DirectionTrackAnchor


def test_upgrade_preserves_initial_prediction() -> None:
    torch.manual_seed(13)
    backbone = SetTransformerDirection(32, 1, 4, 2, 0.0, "engineered")
    model = DirectionTrackAnchor(backbone, 32, 0.0, 50.0).eval()
    x = torch.randn(3, 12, 5)
    mask = torch.ones(3, 12, dtype=torch.bool)
    mask[1, 8:] = False
    x[1, 8:] = 0.0
    with torch.no_grad():
        before_direction, before_point = model(x, mask)
        upgrade_track_model_physics_features(model)
        model.backbone.set_feature_norm(
            torch.tensor([1.0, -43.0, 0.0, 0.0, 20.0]),
            torch.tensor([22.0, 370.0, 39.0, 39.0, 147.0]),
        )
        after_direction, after_point = model(x, mask)
    torch.testing.assert_close(after_direction, before_direction, rtol=0, atol=1e-6)
    torch.testing.assert_close(after_point, before_point, rtol=0, atol=1e-6)


def test_physical_axis_orients_with_time() -> None:
    torch.manual_seed(7)
    backbone = SetTransformerDirection(16, 1, 4, 2, 0.0, "raw")
    model = DirectionTrackAnchor(backbone, 16, 0.0, 50.0)
    upgrade_track_model_physics_features(model)
    model.backbone.set_feature_norm(torch.zeros(5), torch.ones(5))
    x = torch.zeros(1, 10, 5)
    x[0, :, 1] = torch.arange(10).float() * 10
    x[0, :, 4] = torch.arange(10).float() * 3
    mask = torch.ones(1, 10, dtype=torch.bool)
    physical = model.backbone.physical_features(x, mask)
    assert torch.isfinite(physical).all()
    assert physical[0, 0, -1] > 0.99
    assert physical[0, 0, 0] < physical[0, -1, 0]
