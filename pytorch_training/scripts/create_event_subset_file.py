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


def contiguous_ranges(indices: np.ndarray) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    starts = np.r_[0, np.flatnonzero(np.diff(indices) != 1) + 1]
    ends = np.r_[starts[1:], len(indices)]
    return [(int(indices[s]), int(indices[e - 1]) + 1) for s, e in zip(starts, ends, strict=False)]


def select_first_events(n_events: int, count: int) -> np.ndarray:
    if count > n_events:
        raise RuntimeError(f"Requested {count} events, found only {n_events}")
    return np.arange(count, dtype=np.int64)


def select_by_prefixes(
    ev_ids: np.ndarray, prefixes: list[bytes], events_per_type: int
) -> np.ndarray:
    ev_ids = ev_ids.astype("S")
    selected = []
    for prefix in prefixes:
        matches = np.flatnonzero(np.char.startswith(ev_ids, prefix))
        if len(matches) < events_per_type:
            raise RuntimeError(
                f"Requested {events_per_type} events for {prefix.decode()}, found {len(matches)}"
            )
        selected.append(matches[:events_per_type])
        print(f"{prefix.decode()}: selected {events_per_type} / available {len(matches)}")
    return np.sort(np.concatenate(selected)).astype(np.int64)


def copy_event_dataset(src_split: h5.Group, dst_split: h5.Group, name: str, indices: np.ndarray):
    src_path = f"{name}/data"
    if src_path not in src_split:
        return
    src_ds = src_split[src_path]
    dst_ds = ensure_group(dst_split, name).create_dataset(
        "data", shape=(len(indices),) + src_ds.shape[1:], dtype=src_ds.dtype
    )
    pos = 0
    for start, end in contiguous_ranges(indices):
        n = end - start
        dst_ds[pos : pos + n] = src_ds[start:end]
        pos += n


def copy_hit_dataset(
    src_split: h5.Group,
    dst_split: h5.Group,
    name: str,
    indices: np.ndarray,
    ev_starts: np.ndarray,
    adjusted_ev_starts: np.ndarray,
):
    src_path = f"{name}/data"
    if src_path not in src_split:
        return
    src_ds = src_split[src_path]
    total_hits = int(adjusted_ev_starts[-1])
    dst_ds = ensure_group(dst_split, name).create_dataset(
        "data", shape=(total_hits,) + src_ds.shape[1:], dtype=src_ds.dtype
    )

    pos = 0
    for event_start, event_end in contiguous_ranges(indices):
        hit_start = int(ev_starts[event_start])
        hit_end = int(ev_starts[event_end])
        n_hits = hit_end - hit_start
        dst_ds[pos : pos + n_hits] = src_ds[hit_start:hit_end]
        pos += n_hits


def write_split(src: h5.File, dst: h5.File, split: str, indices: np.ndarray):
    src_split = src[split]
    dst_split = ensure_group(dst, split)
    ev_starts = np.asarray(src_split["ev_starts/data"], dtype=np.int64)
    event_sizes = ev_starts[indices + 1] - ev_starts[indices]
    adjusted_ev_starts = np.zeros(len(indices) + 1, dtype=ev_starts.dtype)
    np.cumsum(event_sizes, out=adjusted_ev_starts[1:])
    ensure_group(dst_split, "ev_starts").create_dataset("data", data=adjusted_ev_starts)

    print(
        f"{split}: writing {len(indices)} events, {int(adjusted_ev_starts[-1])} hits, "
        f"{len(contiguous_ranges(indices))} event ranges"
    )
    for name in EVENT_DATASETS:
        copy_event_dataset(src_split, dst_split, name, indices)
    for name in HIT_DATASETS:
        copy_hit_dataset(src_split, dst_split, name, indices, ev_starts, adjusted_ev_starts)


def create_subset(
    source: str,
    target: str,
    train_events: int,
    eval_events_per_type: int,
    prefixes: list[str],
):
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    byte_prefixes = [prefix.encode() for prefix in prefixes]

    with h5.File(source, "r") as src, h5.File(target_path, "w") as dst:
        if "norm_param" in src:
            src.copy("norm_param", dst)

        train_indices = select_first_events(len(src["train/ev_starts/data"]) - 1, train_events)
        write_split(src, dst, "train", train_indices)

        for split in ("val", "test"):
            ev_ids = np.asarray(src[f"{split}/ev_ids/data"])
            indices = select_by_prefixes(ev_ids, byte_prefixes, eval_events_per_type)
            write_split(src, dst, split, indices)

    print(f"Wrote {target_path}")


def main():
    parser = argparse.ArgumentParser(description="Create an event subset HDF5 file.")
    parser.add_argument(
        "--source",
        default="/home2/ivkhar/Baikal/data/normed/baikal_2020_sig-noise_mid-eq_normed.h5",
    )
    parser.add_argument(
        "--target",
        default="/home/plotnikovgp/baikal/data/baikal_2020_sig-noise_mid-eq_normed_5gb.h5",
    )
    parser.add_argument("--train-events", type=int, default=2_000_000)
    parser.add_argument("--eval-events-per-type", type=int, default=25_000)
    parser.add_argument("--prefixes", nargs="+", default=["muatm", "nuatm", "nue2"])
    args = parser.parse_args()

    create_subset(
        source=args.source,
        target=args.target,
        train_events=args.train_events,
        eval_events_per_type=args.eval_events_per_type,
        prefixes=args.prefixes,
    )


if __name__ == "__main__":
    main()
