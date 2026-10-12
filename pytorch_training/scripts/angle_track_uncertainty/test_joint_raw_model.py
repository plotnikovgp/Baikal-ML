"""Joint hit/angle/point model preserves padding invariance and gradients."""

import torch
from joint_raw_model import JointRawReconstructor


def test_joint_forward_padding_and_gradients() -> None:
    torch.manual_seed(21)
    model = JointRawReconstructor(hidden=16, layers=1, ff_size=32, legacy_float_mask=False).eval()
    model.set_feature_norm(torch.zeros(5), torch.ones(5))
    x = torch.randn(2, 8, 5)
    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[1, 5:] = False
    x[1, 5:] = 0
    direction, point, center, logit, probability = model(x, mask)
    assert direction.shape == (2, 3)
    assert point.shape == (2, 3)
    assert center.shape == (2, 3)
    assert logit.shape == probability.shape == (2, 8)
    torch.testing.assert_close(direction.norm(dim=1), torch.ones(2), atol=1e-6, rtol=0)
    torch.testing.assert_close((direction * point).sum(dim=1), torch.zeros(2), atol=1e-6, rtol=0)
    altered = x.clone()
    altered[1, 5:] = 1000
    direction_new, point_new, center_new, _logit, _prob = model(altered, mask)
    torch.testing.assert_close(direction_new[1], direction[1], atol=1e-6, rtol=0)
    torch.testing.assert_close(point_new[1], point[1], atol=1e-6, rtol=0)
    torch.testing.assert_close(center_new[1], center[1], atol=1e-6, rtol=0)
    (direction.sum() + point.sum() + logit[mask].sum()).backward()
    assert model.first_layer.weight.grad is not None
    assert torch.isfinite(model.first_layer.weight.grad).all()
