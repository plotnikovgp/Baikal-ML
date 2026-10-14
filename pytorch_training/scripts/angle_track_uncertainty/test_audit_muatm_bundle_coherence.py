"""Small exact-ID fixture for the raw muatm bundle audit."""

import json

import h5py
import numpy as np
from audit_muatm_bundle_coherence import analyze, direction


def test_direction_axis_and_raw_join(tmp_path):
    np.testing.assert_allclose(
        direction(np.array([np.pi / 2]), np.array([0]))[0], [1, 0, 0], atol=1e-12
    )
    raw = tmp_path / "raw.h5"
    manifest = tmp_path / "manifest.npz"
    report = tmp_path / "report.json"
    ids = np.asarray([b"muatm_1_0", b"muatm_1_1"], dtype="S64")
    prime = np.asarray([[90.0, 0.0, 100.0], [90.0, 0.0, 200.0]])
    muons = np.asarray(
        [
            [np.pi / 2, 0, 0, 0, 0, 0, 10],
            [np.pi / 2, 0, 0, 0, 0, 0, 10],
            [np.pi / 2, 0, 0, 10, 0, 0, 10],
        ],
        dtype=np.float64,
    )
    with h5py.File(raw, "w") as file:
        root = file.create_group("muatm_2020")
        for name, data in (
            ("muons_prty/mu_starts/part_1/data", np.asarray([0, 1, 3])),
            ("muons_prty/individ/part_1/data", muons),
            ("prime_prty/part_1/data", prime),
            ("ev_ids/part_1/data", ids),
        ):
            root.create_dataset(name, data=data)
    np.savez_compressed(
        manifest,
        event_ids=ids,
        part=np.asarray([b"1", b"1"]),
        source_index=np.asarray([0, 1]),
        prime=prime.astype(np.float32),
        anchor_valid=np.asarray([True, False]),
    )
    analyze(manifest, raw, report)
    result = json.loads(report.read_text())
    assert result["muon_multiplicity"]["one"] == 1
    assert result["muon_multiplicity"]["two_to_four"] == 1
    multi = result["groups"]["multi_all"]
    assert multi["max_individual_muon_to_primary_deg"] < 1e-4
    assert multi["transverse_start_spread_q68_m"]["q50"] == 5.0


def test_raw_join_rejects_mismatched_id(tmp_path):
    raw = tmp_path / "raw.h5"
    manifest = tmp_path / "manifest.npz"
    with h5py.File(raw, "w") as file:
        root = file.create_group("muatm_2020")
        for name, data in (
            ("muons_prty/mu_starts/part_1/data", np.asarray([0, 1])),
            ("muons_prty/individ/part_1/data", np.asarray([[np.pi / 2, 0, 0, 0, 0, 0, 1]])),
            ("prime_prty/part_1/data", np.asarray([[90, 0, 100]])),
            ("ev_ids/part_1/data", np.asarray([b"muatm_1_0"], dtype="S64")),
        ):
            root.create_dataset(name, data=data)
    np.savez_compressed(
        manifest,
        event_ids=np.asarray([b"muatm_1_wrong"], dtype="S64"),
        part=np.asarray([b"1"]),
        source_index=np.asarray([0]),
        prime=np.asarray([[90, 0, 100]], dtype=np.float32),
        anchor_valid=np.asarray([True]),
    )
    try:
        analyze(manifest, raw, tmp_path / "report.json")
    except ValueError as error:
        assert "Raw event ID/index mismatch" in str(error)
    else:
        raise AssertionError("Expected an exact-ID join failure")
