"""Unit checks for signed angular residuals in the reco precision audit."""

import numpy as np
from audit_reco_precision_bias import angular_error, unit, wrap_deg


def test_wrap_respects_azimuth_boundary() -> None:
    np.testing.assert_allclose(wrap_deg(np.array([181, -181, 359, -359])), [-179, 179, -1, 1])


def test_constant_correction_recovers_direction() -> None:
    theta = np.array([30.0, 70.0, 110.0])
    phi = np.array([359.0, 10.0, 270.0])
    true_direction = unit(theta, phi)
    biased = angular_error(theta + 0.2, phi + 0.4, true_direction)
    corrected = angular_error(theta, phi, true_direction)
    assert np.all(biased > 0.1)
    np.testing.assert_allclose(corrected, 0.0, atol=2e-6)
