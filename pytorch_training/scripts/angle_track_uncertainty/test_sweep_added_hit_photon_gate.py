"""Input validation for the extended photon-residual gate sweep."""

import numpy as np
import pytest
from sweep_added_hit_photon_gate import validate_photon_thresholds


def test_thresholds_are_increasing_and_positive() -> None:
    np.testing.assert_array_equal(validate_photon_thresholds([200, 250, 300]), [200, 250, 300])
    for values in ([], [0, 200], [200, 200], [300, 200], [float("nan")]):
        with pytest.raises(ValueError, match="Photon thresholds"):
            validate_photon_thresholds(values)
