"""Point-to-true-line geometry and ID-join regression checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from audit_2019_point_uncertainty import load_line_error
from calibrate_cross_season_track import check_truth_primaries


def test_line_error_uses_transverse_distance_and_aligned_truth(tmp_path):
    csv_path = tmp_path / "predictions.csv"
    truth_path = tmp_path / "truth.npz"
    pd.DataFrame(
        {
            "event_id": ["second", "first"],
            "anchor_x_m": [3.0, 0.0],
            "anchor_y_m": [4.0, 0.0],
            "anchor_z_m": [999.0, 7.0],
            "sigma_anchor_m": [2.0, 1.0],
            "n_hits": [12, 8],
        }
    ).to_csv(csv_path, index=False)
    np.savez_compressed(
        truth_path,
        event_ids=np.asarray([b"first", b"second"]),
        valid=np.asarray([True, True]),
        prime=np.asarray([[0, 0, 1], [0, 0, 2]], dtype=np.float32),
        point_m=np.zeros((2, 3), dtype=np.float32),
    )
    rows, error, sigma, excluded = load_line_error(csv_path, truth_path)
    assert rows.event_id.tolist() == ["second", "first"]
    np.testing.assert_allclose(error, [5.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(sigma, [2.0, 1.0])
    assert excluded == 0
    check_truth_primaries(
        np.asarray(["second", "first"]),
        np.asarray([[0, 0, 2], [0, 0, 1]], dtype=np.float32),
        truth_path,
    )
    with pytest.raises(ValueError, match="primaries disagree"):
        check_truth_primaries(
            np.asarray(["second", "first"]),
            np.asarray([[0, 0, 1], [0, 0, 2]], dtype=np.float32),
            truth_path,
        )
