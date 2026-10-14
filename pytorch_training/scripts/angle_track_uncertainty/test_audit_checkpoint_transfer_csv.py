"""Small numerical checks for paired direction-transfer evaluation."""

import numpy as np
import pandas as pd
from audit_checkpoint_transfer_csv import angle_error, bootstrap


def test_angle_error_uses_degrees_and_full_direction() -> None:
    frame = pd.DataFrame(
        {"direction_x": [0.0, 1.0], "direction_y": [0.0, 0.0], "direction_z": [1.0, 0.0]}
    )
    truth = np.asarray([[0.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(angle_error(frame, truth), [0.0, 90.0])


def test_identical_paired_errors_have_zero_bootstrap_delta() -> None:
    errors = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    parts = np.asarray(["a", "a", "b", "b", "c", "c"])
    result = bootstrap(errors, errors, parts)
    for mode in result.values():
        for interval in mode.values():
            np.testing.assert_array_equal(interval, [0.0, 0.0])
