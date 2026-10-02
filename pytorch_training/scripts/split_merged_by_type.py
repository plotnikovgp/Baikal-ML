"""Split merged MC h5 dataset into per-type h5 files (muatm, nuatm, nue2).

Each output file preserves the same normalization and has train/val groups
with data, labels, t_res, ev_starts, channels, ev_ids.
"""

from pathlib import Path

import h5py
import numpy as np

MERGED_PATH = "/home/plotnikovgp/baikal/data/baikal_mc_merged_m50-na15-nue15_all_smaller_norm.h5"
OUT_DIR = Path("/home/plotnikovgp/baikal/data")
TYPES = ["muatm", "nuatm", "nue2"]


def get_event_type(ev_id: str) -> str:
    return ev_id.split("_")[0]


def extract_type_events(src, split: str, ev_type: str):
    """Extract events of a given type from a split, return dict of arrays."""
    ev_ids_raw = src[f"{split}/ev_ids/data"][:]
    ev_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in ev_ids_raw]

    mask = np.array([get_event_type(eid) == ev_type for eid in ev_ids])
    indices = np.where(mask)[0]
    print(f"  {split}/{ev_type}: {len(indices)} events out of {len(ev_ids)}")

    if len(indices) == 0:
        return None

    ev_starts_all = src[f"{split}/ev_starts/data"][:]
    data_all = src[f"{split}/data/data"]
    labels_all = src[f"{split}/labels/data"]
    t_res_all = src[f"{split}/t_res/data"]
    channels_all = src[f"{split}/channels/data"]
    ev_ids_arr = ev_ids_raw

    new_data = []
    new_labels = []
    new_t_res = []
    new_channels = []
    new_ev_ids = []
    new_ev_starts = [0]

    for idx in indices:
        start = ev_starts_all[idx]
        end = ev_starts_all[idx + 1]
        n_hits = end - start

        new_data.append(data_all[start:end])
        new_labels.append(labels_all[start:end])
        new_t_res.append(t_res_all[start:end])
        new_channels.append(channels_all[start:end])
        new_ev_ids.append(ev_ids_arr[idx])
        new_ev_starts.append(new_ev_starts[-1] + n_hits)

    return {
        "data": np.concatenate(new_data, axis=0),
        "labels": np.concatenate(new_labels, axis=0),
        "t_res": np.concatenate(new_t_res, axis=0),
        "channels": np.concatenate(new_channels, axis=0),
        "ev_ids": np.array(new_ev_ids),
        "ev_starts": np.array(new_ev_starts, dtype=np.int64),
    }


def write_split(dst, split: str, arrays: dict):
    grp = dst.create_group(split)
    grp.create_dataset("data/data", data=arrays["data"])
    grp.create_dataset("labels/data", data=arrays["labels"])
    grp.create_dataset("t_res/data", data=arrays["t_res"])
    grp.create_dataset("channels/data", data=arrays["channels"])
    grp.create_dataset("ev_ids/data", data=arrays["ev_ids"])
    grp.create_dataset("ev_starts/data", data=arrays["ev_starts"])


def main():
    src = h5py.File(MERGED_PATH, "r")
    norm_mean = src["norm_param/mean"][:]
    norm_std = src["norm_param/std"][:]

    for ev_type in TYPES:
        out_path = OUT_DIR / f"merged_{ev_type}.h5"
        print(f"\nCreating {out_path}")
        dst = h5py.File(out_path, "w")

        norm_grp = dst.create_group("norm_param")
        norm_grp.create_dataset("mean", data=norm_mean)
        norm_grp.create_dataset("std", data=norm_std)

        for split in ["train", "val", "test"]:
            arrays = extract_type_events(src, split, ev_type)
            if arrays is not None:
                write_split(dst, split, arrays)

        dst.close()
        print(f"  Done: {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")

    src.close()
    print("\nAll done.")


if __name__ == "__main__":
    main()
