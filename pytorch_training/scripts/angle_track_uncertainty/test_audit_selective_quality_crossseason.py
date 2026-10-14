"""Small alignment and primary-exclusion checks for selective-quality audit."""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest
from audit_selective_quality_crossseason import joined, load_p070, novel_2020


def test_load_p070_joins_truth_by_id_not_csv_row(tmp_path):
    h5_path = tmp_path / "events.h5"
    with h5py.File(h5_path, "w") as h5:
        test = h5.create_group("test")
        test.create_dataset("ev_ids/data", data=np.array([b"a", b"b"]))
        test.create_dataset(
            "prime_prty/data",
            data=np.array([[10, 20, 100], [30, 40, 200]], dtype=np.float32),
        )
    csv_path = tmp_path / "pred.csv"
    pd.DataFrame(
        {
            "event_id": ["b", "a"],
            "theta_deg": [31, 11],
            "phi_deg": [41, 21],
            "n_hits": [9, 8],
            "sigma_angle_deg": [2.0, 1.0],
        }
    ).to_csv(csv_path, index=False)
    rows = load_p070(csv_path, h5_path)
    np.testing.assert_array_equal(rows["truth"], [[30, 40], [10, 20]])
    np.testing.assert_array_equal(rows["sigma"], [2.0, 1.0])


def test_joined_rejects_hit_topology_mismatch(tmp_path):
    features = tmp_path / "features.csv"
    predictions = tmp_path / "pred.csv"
    pd.DataFrame({"event_id": ["a"], "n_hits": [8], "n_strings": [2]}).to_csv(features, index=False)
    pd.DataFrame(
        {
            "event_id": ["a"],
            "n_hits": [9],
            "n_strings": [2],
            "theta_deg": [50],
            "sigma_angle_deg": [2],
        }
    ).to_csv(predictions, index=False)
    with pytest.raises(ValueError, match="topology mismatch"):
        joined(features, predictions, deduplicate=False)


def test_novel_2020_excludes_every_reference_split(tmp_path):
    ref_gt = tmp_path / "ref_gt.h5"
    ref_pred = tmp_path / "ref_pred.h5"
    known = np.array([[10, 20, 100]], dtype=np.float32)
    for path in (ref_gt, ref_pred):
        with h5py.File(path, "w") as h5:
            for split in ("train", "val", "test"):
                h5.create_dataset(f"{split}/prime_prty/data", data=known)
    data = tmp_path / "novel.h5"
    with h5py.File(data, "w") as h5:
        h5.create_dataset("train/ev_ids/data", data=np.array([b"known", b"new1"]))
        h5.create_dataset(
            "train/prime_prty/data",
            data=np.array([[10, 20, 100], [30, 40, 200]], dtype=np.float32),
        )
        h5.create_dataset("val/ev_ids/data", data=np.array([b"new2"]))
        h5.create_dataset("val/prime_prty/data", data=np.array([[50, 60, 300]], dtype=np.float32))
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    pd.DataFrame(
        {
            "event_id": ["known", "new1"],
            "theta_deg": [10, 30],
            "phi_deg": [20, 40],
            "sigma_angle_deg": [1.0, 2.0],
        }
    ).to_csv(train_csv, index=False)
    pd.DataFrame(
        {"event_id": ["new2"], "theta_deg": [50], "phi_deg": [60], "sigma_angle_deg": [3.0]}
    ).to_csv(val_csv, index=False)
    result = novel_2020(data, ref_gt, ref_pred, train_csv, val_csv)
    assert len(result["error"]) == 2
    np.testing.assert_allclose(result["error"], 0, atol=1e-4)
    np.testing.assert_array_equal(result["sigma"], [2, 3])
