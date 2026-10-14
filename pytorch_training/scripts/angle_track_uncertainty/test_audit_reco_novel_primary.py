"""Unit tests for the primary-disjoint reco comparison."""

import numpy as np
import pytest
from audit_reco_novel_primary import (
    angular_error_deg,
    bootstrap_unique_primary_delta,
    summarize,
    vectors_from_angles,
)


def test_angular_error_wraps_phi_and_normalizes_direction() -> None:
    target = vectors_from_angles(np.asarray([[90.0, 359.0], [0.0, 50.0]]))
    prediction = vectors_from_angles(np.asarray([[90.0, 1.0], [0.0, 200.0]])) * 3
    np.testing.assert_allclose(angular_error_deg(prediction, target), [2.0, 0.0], atol=1e-5)
    with pytest.raises(ValueError, match="zero predicted"):
        angular_error_deg(np.zeros((1, 3)), target[:1])


def test_summary_and_primary_bootstrap() -> None:
    keys = np.asarray([b"a", b"a", b"b", b"c"])
    first = np.asarray([1.0, 2.0, 3.0, 4.0])
    second = first + 1
    radius68 = np.asarray([1.0, 1.0, 3.0, 3.0])
    radius95 = np.full(4, 10.0)
    report = summarize(
        np.asarray([True, False, True, True]),
        first,
        second,
        keys,
        first_label="nn",
        second_label="reco",
        radius68=radius68,
        radius95=radius95,
    )
    assert report["count"] == 3
    assert report["unique_primaries"] == 3
    assert report["fraction_first_better_eventwise"] == 1.0
    assert report["angle_coverage68"] == 2 / 3
    assert report["angle_coverage95"] == 1.0
    assert summarize(
        np.zeros(4, dtype=bool), first, second, keys, first_label="a", second_label="b"
    ) == {"count": 0}
    bootstrap = bootstrap_unique_primary_delta(keys, first, second)
    assert bootstrap["unique_primaries"] == 3
    np.testing.assert_allclose(bootstrap["delta_first_minus_second_ci95"]["q50"], [-1, -1])
    np.testing.assert_allclose(bootstrap["delta_first_minus_second_ci95"]["q68"], [-1, -1])
