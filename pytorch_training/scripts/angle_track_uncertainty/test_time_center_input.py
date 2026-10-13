"""The optional time-invariance diagnostic must preserve the default input."""

import h5py
import numpy as np
from infer_rethresholded_for_audit import selected_batch


def test_selected_batch_center_time_alpha(tmp_path):
    source = tmp_path / "one_event.h5"
    values = np.zeros((8, 5), dtype=np.float32)
    values[:, 1] = np.arange(8, dtype=np.float32)
    with h5py.File(source, "w") as h5:
        group = h5.create_group("test")
        group.create_dataset("ev_starts/data", data=np.asarray([0, 8], dtype=np.int64))
        group.create_dataset("data/data", data=values)
        group.create_dataset("signal_probability/data", data=np.full(8, 0.9))
        group.create_dataset("channels/data", data=np.asarray([0, 1, 2, 3, 36, 37, 38, 39]))
        group.create_dataset("ev_ids/data", data=np.asarray([b"event"], dtype="S5"))
    prep = {
        "input_norm_mean": [0.0] * 5,
        "input_norm_std": [1.0] * 5,
        "max_seq_len": 128,
        "center_time": False,
    }
    with h5py.File(source, "r") as h5:
        args = (h5["test"], 0, 1, 0.7, prep, np.zeros(5), np.ones(5), 36, 0, 1)
        kept, original, mask = selected_batch(*args)
        centered_rows, centered, centered_mask = selected_batch(*args, center_time_alpha=1.0)
    assert len(kept) == len(centered_rows) == 1
    np.testing.assert_array_equal(mask, centered_mask)
    np.testing.assert_array_equal(original[0, :, 1], np.arange(8))
    np.testing.assert_allclose(centered[0, :, 1], np.arange(8) - 3.5)
    np.testing.assert_array_equal(centered[..., (0, 2, 3, 4)], original[..., (0, 2, 3, 4)])
