"""Build h5 with nuatm+nue2 events that pass 2-8 cut on GT signal (label!=0).

Keeps only signal hits per event. Output has train/val splits with
data, labels, t_res, channels, ev_starts, ev_ids.
"""

from pathlib import Path

import h5py
import numpy as np

SRC = "/home/plotnikovgp/baikal/data/baikal_mc_merged_m50-na15-nue15_all_smaller_norm.h5"
OUT = "/home/plotnikovgp/baikal/data/merged_nue2_nuatm_2-8gt_sigonly.h5"

MIN_HITS = 8
MIN_STRINGS = 2
TYPES_KEEP = {"nuatm", "nue2"}


def process_split(src, split: str):
    ev_starts = src[f"{split}/ev_starts/data"][:]
    ev_ids_raw = src[f"{split}/ev_ids/data"][:]
    n_events = len(ev_starts) - 1

    out_data = []
    out_labels = []
    out_tres = []
    out_channels = []
    out_ev_ids = []
    out_ev_starts = [0]

    kept = 0
    for i in range(n_events):
        eid = ev_ids_raw[i]
        eid_str = eid.decode() if isinstance(eid, bytes) else str(eid)
        etype = eid_str.split("_")[0]
        if etype not in TYPES_KEEP:
            continue

        s, e = int(ev_starts[i]), int(ev_starts[i + 1])
        labels = src[f"{split}/labels/data"][s:e]
        channels = src[f"{split}/channels/data"][s:e]

        sig = labels != 0
        n_sig = int(sig.sum())
        if n_sig < MIN_HITS:
            continue
        n_strings = len(np.unique(channels[sig] // 36))
        if n_strings < MIN_STRINGS:
            continue

        data = src[f"{split}/data/data"][s:e].astype(np.float32)
        t_res = src[f"{split}/t_res/data"][s:e].astype(np.float32)

        out_data.append(data[sig])
        out_labels.append(labels[sig])
        out_tres.append(t_res[sig])
        out_channels.append(channels[sig])
        out_ev_ids.append(eid)
        out_ev_starts.append(out_ev_starts[-1] + n_sig)
        kept += 1

        if kept % 50000 == 0:
            print(f"  {split}: kept {kept} events so far...")

    print(f"  {split}: {kept}/{n_events} events kept")
    if kept == 0:
        return None

    return {
        "data": np.concatenate(out_data, axis=0),
        "labels": np.concatenate(out_labels, axis=0),
        "t_res": np.concatenate(out_tres, axis=0),
        "channels": np.concatenate(out_channels, axis=0),
        "ev_ids": np.array(out_ev_ids),
        "ev_starts": np.array(out_ev_starts, dtype=np.int64),
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
    src = h5py.File(SRC, "r")
    dst = h5py.File(OUT, "w")

    norm_grp = dst.create_group("norm_param")
    norm_grp.create_dataset("mean", data=src["norm_param/mean"][:])
    norm_grp.create_dataset("std", data=src["norm_param/std"][:])

    for split in ["train", "val", "test"]:
        print(f"Processing {split}...")
        arrays = process_split(src, split)
        if arrays is not None:
            write_split(dst, split, arrays)

    src.close()
    dst.close()
    p = Path(OUT)
    print(f"\nDone: {OUT} ({p.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
