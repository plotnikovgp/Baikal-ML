import argparse
import json
import random
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


def select_parts(
    src: h5.File, particle: str, train_parts: int, val_parts: int, test_parts: int, seed: int
):
    rng = random.Random(seed)
    parts = list(src[f"{particle}/data"].keys())
    rng.shuffle(parts)
    needed = train_parts + val_parts + test_parts
    if len(parts) < needed:
        raise RuntimeError(f"Requested {needed} parts for {particle}, found {len(parts)}")
    return {
        "train": parts[:train_parts],
        "val": parts[train_parts : train_parts + val_parts],
        "test": parts[train_parts + val_parts : needed],
    }


def compute_norm(src: h5.File, particle: str, parts: list[str], q_upper_bound: float | None):
    total = 0
    sum_x = None
    sum_x2 = None
    for part in parts:
        data = np.asarray(src[f"{particle}/data/{part}/data"], dtype=np.float64)
        if q_upper_bound is not None:
            data[:, 0] = np.minimum(data[:, 0], q_upper_bound)
        if sum_x is None:
            sum_x = np.zeros(data.shape[1:], dtype=np.float64)
            sum_x2 = np.zeros(data.shape[1:], dtype=np.float64)
        total += data.shape[0]
        sum_x += data.sum(axis=0, dtype=np.float64)
        sum_x2 += np.square(data, dtype=np.float64).sum(axis=0, dtype=np.float64)
    mean = sum_x / total
    var = np.maximum(sum_x2 / total - mean**2, 0.0)
    std = np.sqrt(var)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def split_sizes(src: h5.File, particle: str, parts: list[str]):
    events = 0
    hits = 0
    for part in parts:
        events += src[f"{particle}/ev_starts/{part}/data"].shape[0] - 1
        hits += src[f"{particle}/data/{part}/data"].shape[0]
    return events, hits


def create_output_datasets(src: h5.File, dst: h5.File, particle: str, split: str, parts: list[str]):
    n_events, n_hits = split_sizes(src, particle, parts)
    dst_split = ensure_group(dst, split)
    datasets = {}

    ensure_group(dst_split, "ev_starts")
    datasets["ev_starts"] = dst_split["ev_starts"].create_dataset(
        "data", shape=(n_events + 1,), dtype=np.int64, chunks=True
    )
    datasets["ev_starts"][0] = 0

    sample_part = parts[0]
    for name in EVENT_DATASETS:
        src_path = f"{particle}/{name}/{sample_part}/data"
        if src_path not in src:
            continue
        src_ds = src[src_path]
        datasets[name] = ensure_group(dst_split, name).create_dataset(
            "data", shape=(n_events,) + src_ds.shape[1:], dtype=src_ds.dtype, chunks=True
        )

    for name in HIT_DATASETS:
        src_path = f"{particle}/{name}/{sample_part}/data"
        if src_path not in src:
            continue
        src_ds = src[src_path]
        dtype = np.float32 if name == "data" else src_ds.dtype
        datasets[name] = ensure_group(dst_split, name).create_dataset(
            "data", shape=(n_hits,) + src_ds.shape[1:], dtype=dtype, chunks=True
        )
    return datasets


def write_split(
    src: h5.File,
    dst: h5.File,
    particle: str,
    split: str,
    parts: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    q_upper_bound: float | None,
):
    datasets = create_output_datasets(src, dst, particle, split, parts)
    event_pos = 0
    hit_pos = 0
    print(f"{split}: writing {len(parts)} parts")

    for part in parts:
        ev_starts = np.asarray(src[f"{particle}/ev_starts/{part}/data"], dtype=np.int64)
        event_sizes = np.diff(ev_starts)
        n_events = len(event_sizes)
        n_hits = int(event_sizes.sum())

        datasets["ev_starts"][event_pos + 1 : event_pos + n_events + 1] = hit_pos + np.cumsum(
            event_sizes, dtype=np.int64
        )

        for name in EVENT_DATASETS:
            src_path = f"{particle}/{name}/{part}/data"
            if name in datasets and src_path in src:
                datasets[name][event_pos : event_pos + n_events] = src[src_path][()]

        for name in HIT_DATASETS:
            src_path = f"{particle}/{name}/{part}/data"
            if name not in datasets or src_path not in src:
                continue
            data = src[src_path][()]
            if name == "data":
                data = data.astype(np.float32, copy=False)
                if q_upper_bound is not None:
                    data[:, 0] = np.minimum(data[:, 0], q_upper_bound)
                data = (data - mean) / std
            datasets[name][hit_pos : hit_pos + n_hits] = data

        event_pos += n_events
        hit_pos += n_hits

    print(f"{split}: events={event_pos}, hits={hit_pos}")


def create_subset(
    source: str,
    target: str,
    particle: str,
    train_parts: int,
    val_parts: int,
    test_parts: int,
    seed: int,
    q_upper_bound: float | None,
):
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with h5.File(source, "r") as src:
        selected = select_parts(src, particle, train_parts, val_parts, test_parts, seed)
        for split, parts in selected.items():
            n_events, n_hits = split_sizes(src, particle, parts)
            print(f"{split}: selected {len(parts)} parts, events={n_events}, hits={n_hits}")

        mean, std = compute_norm(src, particle, selected["train"], q_upper_bound)
        print("mean", mean.tolist())
        print("std", std.tolist())

        with h5.File(target_path, "w") as dst:
            for split in ("train", "val", "test"):
                write_split(
                    src=src,
                    dst=dst,
                    particle=particle,
                    split=split,
                    parts=selected[split],
                    mean=mean,
                    std=std,
                    q_upper_bound=q_upper_bound,
                )
            ensure_group(dst, "norm_param").create_dataset("mean", data=mean, dtype=np.float32)
            dst["norm_param"].create_dataset("std", data=std, dtype=np.float32)

    config_path = target_path.with_suffix(".config")
    config_path.write_text(
        json.dumps(
            {
                "source": source,
                "target": target,
                "particle": particle,
                "train_parts": train_parts,
                "val_parts": val_parts,
                "test_parts": test_parts,
                "seed": seed,
                "q_upper_bound": q_upper_bound,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Wrote {target_path}")
    print(f"Wrote {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a normalized subset from particle-part HDF5."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--particle", default="muatm")
    parser.add_argument("--train-parts", type=int, default=120)
    parser.add_argument("--val-parts", type=int, default=10)
    parser.add_argument("--test-parts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--q-upper-bound", type=float, default=None)
    args = parser.parse_args()

    create_subset(
        source=args.source,
        target=args.target,
        particle=args.particle,
        train_parts=args.train_parts,
        val_parts=args.val_parts,
        test_parts=args.test_parts,
        seed=args.seed,
        q_upper_bound=args.q_upper_bound,
    )


if __name__ == "__main__":
    main()
