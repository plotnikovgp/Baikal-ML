"""Validate duplicate-ID handling for the MC multiplicity-domain audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from audit_muatm_domain_composition import align_anchor_validity


def test_consistent_repeated_anchor_ids_join_to_unique_domain_ids():
    ids = np.array([b"muatm_a", b"muatm_a", b"muatm_b", b"nue2_c"])
    valid = np.array([True, True, False, True])
    source = np.array([1, 1, 2, 3])
    domain_ids = pd.Index(["muatm_b", "muatm_a"])
    np.testing.assert_array_equal(
        align_anchor_validity(ids, valid, source, domain_ids), [False, True]
    )


def test_repeated_anchor_id_with_conflicting_source_is_rejected():
    ids = np.array([b"muatm_a", b"muatm_a"])
    valid = np.array([True, True])
    source = np.array([1, 2])
    with pytest.raises(ValueError, match="Repeated anchor event IDs disagree"):
        align_anchor_validity(ids, valid, source, pd.Index(["muatm_a"]))


def test_missing_domain_id_is_rejected():
    ids = np.array([b"muatm_a"])
    with pytest.raises(ValueError, match="missing from muatm anchors"):
        align_anchor_validity(ids, np.array([True]), np.array([1]), pd.Index(["muatm_missing"]))
