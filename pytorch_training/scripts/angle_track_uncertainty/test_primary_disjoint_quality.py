"""Training-primary overlap must be removed from an independent MC audit."""

import numpy as np
from audit_timegate_mc_quality import novel_primary_mask


def test_novel_primary_mask_uses_union_of_both_training_streams():
    predicted_train = np.array([[1, 2, 3, 0, 0, 0], [4, 5, 6, 0, 0, 0]], dtype=np.float32)
    gt_train = np.array([[7, 8, 9, 0, 0, 0]], dtype=np.float32)
    test = np.array(
        [[1, 2, 3, 99, 99, 99], [7, 8, 9, 0, 0, 0], [10, 11, 12, 0, 0, 0]],
        dtype=np.float32,
    )
    novel, count = novel_primary_mask(test, predicted_train, gt_train)
    assert count == 3
    np.testing.assert_array_equal(novel, [False, False, True])
