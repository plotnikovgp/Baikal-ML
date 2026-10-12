"""Regression test for exact-ID paired domain comparisons."""

import pandas as pd
from compare_domain_models_paired import aligned


def test_aligned_reorders_identical_ids(tmp_path):
    before = pd.DataFrame(
        {
            "event_id": ["a", "b"],
            "n_hits": [8, 12],
            "n_strings": [2, 3],
            "theta_deg": [110.0, 120.0],
        }
    )
    after = pd.DataFrame(
        {
            "event_id": ["b", "a"],
            "n_hits": [12, 8],
            "n_strings": [3, 2],
            "theta_deg": [121.0, 111.0],
        }
    )
    before_path = tmp_path / "before.csv"
    after_path = tmp_path / "after.csv"
    before.to_csv(before_path, index=False)
    after.to_csv(after_path, index=False)

    ordered_before, ordered_after = aligned(before_path, after_path, False)
    assert ordered_before.event_id.tolist() == ["a", "b"]
    assert ordered_after.event_id.tolist() == ["a", "b"]
    assert ordered_after.theta_deg.tolist() == [111.0, 121.0]
