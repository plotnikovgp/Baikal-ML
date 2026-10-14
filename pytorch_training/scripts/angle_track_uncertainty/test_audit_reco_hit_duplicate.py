"""Tests for order-independent exact hit fingerprints."""

import h5py
import numpy as np
from audit_reco_hit_duplicate import fingerprint


def test_fingerprint_ignores_hit_order_but_not_hit_values(tmp_path) -> None:
    path = tmp_path / "hits.h5"
    rows = np.asarray(
        [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]],
        dtype=np.float32,
    )
    changed = rows.copy()
    changed[0, 1] += np.float32(0.001)
    with h5py.File(path, "w") as h5:
        group = h5.create_group("test")
        group.create_dataset("ev_starts/data", data=[0, 2, 4, 6])
        group.create_dataset("data/data", data=np.concatenate((rows, rows[::-1], changed)))
        assert fingerprint(group, 0) == fingerprint(group, 1)
        assert fingerprint(group, 0) != fingerprint(group, 2)
