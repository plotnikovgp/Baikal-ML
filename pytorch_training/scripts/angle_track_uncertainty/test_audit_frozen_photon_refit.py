"""Geometry checks for the fixed-parameter Cherenkov-refit audit."""

import numpy as np
from audit_frozen_photon_refit import angle_deg, quantiles, unit


def test_spherical_direction_and_angular_error() -> None:
    directions = unit(np.array([90.0, 90.0]), np.array([0.0, 90.0]))
    np.testing.assert_allclose(directions, [[1, 0, 0], [0, 1, 0]], atol=1e-12)
    np.testing.assert_allclose(angle_deg(directions[0], directions[1]), 90.0)


def test_quantiles() -> None:
    result = quantiles(np.array([0.0, 1.0, 2.0, 3.0]))
    assert result["q50"] == 1.5
    assert result["q68"] == 2.04
