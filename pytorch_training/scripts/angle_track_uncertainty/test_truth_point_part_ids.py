"""Truth-point lookup must support both 2019 and 2020 part ID layouts."""

import h5py
import numpy as np
from build_all_track_anchor_targets import raw_particle_points


def test_truth_point_lookup_accepts_underscore_in_2019_part(tmp_path):
    path = tmp_path / "source.h5"

    def put(root, name, values):
        parent, key = name.rsplit("/", 1)
        root.require_group(parent).create_dataset(key, data=values)

    with h5py.File(path, "w") as source:
        for year, part, event_id in (
            (2019, "1000_4", b"nue2_1000_4_0"),
            (2020, "12", b"nue2_12_0"),
        ):
            root = source.create_group(f"nue2_{year}")
            put(root, "clusters_centers/data", np.asarray([[0, 0, 0], [10, 20, 30]]))
            put(root, "coords_are_cluster_centered/data", np.asarray(True))
            put(root, f"ev_ids/part_{part}/data", np.asarray([event_id]))
            put(root, f"muons_prty/mu_starts/part_{part}/data", np.asarray([0, 1]))
            put(root, f"muons_prty/individ/part_{part}/data", np.asarray([[0, 0, 11, 22, 33]]))
            put(root, f"raw/cluster_ids/part_{part}/data", np.asarray([1]))

    with h5py.File(path) as source:
        for year, event_id in ((2019, b"nue2_1000_4_0"), (2020, b"nue2_12_0")):
            points, parsed, single, indices = raw_particle_points(
                source, f"nue2_{year}", np.asarray([event_id])
            )
            assert parsed.tolist() == [True]
            assert single.tolist() == [True]
            assert indices.tolist() == [0]
            np.testing.assert_allclose(points, [[1, 2, 3]])
