"""Holdout checks must compare source parts, IDs, and generated primaries."""

import json

import h5py
import numpy as np
from audit_independent_holdout_overlap import overlap


def write_sample(path, parts, ids, primaries):
    with h5py.File(path, "w") as h5:
        h5.attrs["metadata_json"] = json.dumps({"selected_parts": parts})
        h5.create_dataset("test/ev_ids/data", data=np.asarray(ids, dtype="S32"))
        h5.create_dataset("test/prime_prty/data", data=np.asarray(primaries, dtype=np.float32))


def test_overlap_detects_each_independent_failure_mode(tmp_path):
    candidate = tmp_path / "candidate.h5"
    reference = tmp_path / "reference.h5"
    write_sample(candidate, ["part_a"], ["id_a", "id_b"], [[1, 2, 3, 0], [4, 5, 6, 0]])
    write_sample(reference, ["part_a"], ["id_a", "id_c"], [[9, 9, 9, 0], [4, 5, 6, 0]])
    result = overlap(candidate, [reference])["references"][str(reference)]
    assert result == {
        "source_part_overlap": 1,
        "event_id_overlap": 1,
        "generated_primary_overlap": 1,
    }
