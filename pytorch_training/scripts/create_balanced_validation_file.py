import argparse
from pathlib import Path

import h5py as h5
import numpy as np

EVENT_DATASETS = ("ev_ids", "num_un_strings", "prime_prty")
HIT_DATASETS = ("data", "labels", "channels", "t_res")


def ensure_group(root: h5.Group, path: str) -> h5.Group:
    group = root
    for part in [p for p in path.split("/") if p]:
        group = group.require_group(part)
    return group


def find_indices(ev_ids: np.ndarray, prefixes: list[bytes], events_per_type: int) -> np.ndarray:
    selected = []
    for prefix in prefixes:
        matches = np.flatnonzero(np.char.startswith(ev_ids.astype("S"), prefix))
        if len(matches) < events_per_type:
            raise RuntimeError(
                f"Requested {events_per_type} events for {prefix.decode()}, found {len(matches)}"
            )
        selected.append(matches[:events_per_type])
        print(f"{prefix.decode()}: selected {events_per_type} / available {len(matches)}")
    return np.sort(np.concatenate(selected)).astype(np.int64)


def write_split(
    src: h5.File, dst: h5.File, source_split: str, target_split: str, indices: np.ndarray
):
    ev_starts = np.asarray(src[f"{source_split}/ev_starts/data"], dtype=np.int64)
    event_sizes = ev_starts[indices + 1] - ev_starts[indices]
    adjusted_ev_starts = np.zeros(len(indices) + 1, dtype=ev_starts.dtype)
    np.cumsum(event_sizes, out=adjusted_ev_starts[1:])
    total_hits = int(adjusted_ev_starts[-1])

    split_group = ensure_group(dst, target_split)
    ensure_group(split_group, "ev_starts").create_dataset("data", data=adjusted_ev_starts)

    for name in EVENT_DATASETS:
        src_path = f"{source_split}/{name}/data"
        if src_path not in src:
            continue
        ensure_group(split_group, name).create_dataset(
            "data", data=np.asarray(src[src_path][indices])
        )

    for name in HIT_DATASETS:
        src_path = f"{source_split}/{name}/data"
        if src_path not in src:
            continue
        src_ds = src[src_path]
        dst_shape = (total_hits,) + src_ds.shape[1:]
        dst_ds = ensure_group(split_group, name).create_dataset(
            "data", shape=dst_shape, dtype=src_ds.dtype
        )
        pos = 0
        for idx, size in zip(indices, event_sizes, strict=False):
            start = int(ev_starts[idx])
            end = int(ev_starts[idx + 1])
            next_pos = pos + int(size)
            dst_ds[pos:next_pos] = src_ds[start:end]
            pos = next_pos


def create_balanced_validation_file(
    source: str,
    target: str,
    source_split: str,
    events_per_type: int,
    prefixes: list[str],
):
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    byte_prefixes = [prefix.encode() for prefix in prefixes]

    with h5.File(source, "r") as src, h5.File(target_path, "w") as dst:
        if "norm_param" in src:
            src.copy("norm_param", dst)

        ev_ids = np.asarray(src[f"{source_split}/ev_ids/data"])
        indices = find_indices(ev_ids, byte_prefixes, events_per_type)

        for target_split in ("val", "test", "train"):
            write_split(src, dst, source_split, target_split, indices)

        print(f"Total events: {len(indices)}")
        print(f"Total hits: {int(dst['val/ev_starts/data'][-1])}")
        print(f"Wrote {target_path}")


def main():
    parser = argparse.ArgumentParser(description="Create a balanced HDF5 validation file.")
    parser.add_argument(
        "--source",
        default="/home2/ivkhar/Baikal/data/normed/baikal_2020_sig-noise_mid-eq_normed.h5",
    )
    parser.add_argument(
        "--target",
        default="/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_2020_balanced_val_2k_each.h5",
    )
    parser.add_argument("--source-split", default="val")
    parser.add_argument("--events-per-type", type=int, default=2000)
    parser.add_argument("--prefixes", nargs="+", default=["muatm", "nue2", "nuatm"])
    args = parser.parse_args()

    create_balanced_validation_file(
        source=args.source,
        target=args.target,
        source_split=args.source_split,
        events_per_type=args.events_per_type,
        prefixes=args.prefixes,
    )


if __name__ == "__main__":
    main()
