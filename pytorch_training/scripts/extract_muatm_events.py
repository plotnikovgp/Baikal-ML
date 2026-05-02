import argparse
from pathlib import Path

import h5py as h5
import numpy as np


def find_event_indices_by_prefix(
    ev_ids_ds, prefix: bytes, max_events: int | None = None
) -> np.ndarray:
    ev_ids = np.array(ev_ids_ds[:], dtype=object)
    mask = np.vectorize(lambda x: x.startswith(prefix))(ev_ids)
    indices = np.nonzero(mask)[0]
    if max_events is not None:
        indices = indices[:max_events]
    return indices.astype(np.int64)


def ensure_group(root: h5.Group, path: str) -> h5.Group:
    parts = [p for p in path.split("/") if p]
    grp = root
    for p in parts:
        if p not in grp:
            grp = grp.create_group(p)
        else:
            grp = grp[p]
    return grp


def copy_norm_params(src: h5.File, dst: h5.File) -> None:
    if "norm_param" in src:
        if "norm_param" in dst:
            del dst["norm_param"]
        src.copy("norm_param", dst)


def extract_events(
    source_path: str,
    target_path: str,
    split: str = "test",
    particle_prefix: bytes = b"nuatm",
    num_events: int = 100_000,
) -> None:
    source_path = str(source_path)
    target_path = str(target_path)

    Path(target_path).parent.mkdir(parents=True, exist_ok=True)

    with h5.File(source_path, "r") as src, h5.File(target_path, "w") as dst:
        copy_norm_params(src, dst)

        ev_ids_path = f"{split}/ev_ids/data"
        ev_starts_path = f"{split}/ev_starts/data"

        if ev_ids_path not in src or ev_starts_path not in src:
            raise KeyError(
                f"Required datasets '{ev_ids_path}' or '{ev_starts_path}' not found in source file"
            )

        ev_ids_ds = src[ev_ids_path]
        ev_starts = np.array(src[ev_starts_path], dtype=np.int64)

        indices = find_event_indices_by_prefix(ev_ids_ds, particle_prefix, max_events=num_events)
        if len(indices) == 0:
            raise RuntimeError(f"No events found with prefix {particle_prefix!r} in {ev_ids_path}")

        if len(indices) < num_events:
            print(f"Warning: requested {num_events} events, but only {len(indices)} are available")

        indices = np.sort(indices)

        event_sizes = ev_starts[indices + 1] - ev_starts[indices]
        adjusted_ev_starts = np.zeros(len(indices) + 1, dtype=np.int64)
        np.cumsum(event_sizes, out=adjusted_ev_starts[1:])
        total_hits = int(adjusted_ev_starts[-1])

        base_group = ensure_group(dst, split)
        ev_starts_dst = ensure_group(base_group, "ev_starts")
        ev_starts_dst.create_dataset("data", data=adjusted_ev_starts.astype(np.int32))

        event_datasets = ["ev_ids", "num_un_strings", "prime_prty"]
        for name in event_datasets:
            src_path = f"{split}/{name}/data"
            if src_path not in src:
                print(f"Skipping missing per-event dataset {src_path}")
                continue
            dst_group = ensure_group(base_group, name)
            data = np.array(src[src_path][indices])
            dst_group.create_dataset("data", data=data)

        hit_datasets = ["data", "labels", "channels", "t_res"]
        for name in hit_datasets:
            src_path = f"{split}/{name}/data"
            if src_path not in src:
                print(f"Skipping missing per-hit dataset {src_path}")
                continue

            src_ds = src[src_path]
            dtype = src_ds.dtype
            shape_tail = src_ds.shape[1:] if len(src_ds.shape) > 1 else ()
            dst_shape = (total_hits,) + shape_tail

            dst_group = ensure_group(base_group, name)
            dst_ds = dst_group.create_dataset("data", shape=dst_shape, dtype=dtype)

            pos = 0
            for idx, size in zip(indices, event_sizes, strict=False):
                start_src = int(ev_starts[idx])
                end_src = int(ev_starts[idx + 1])
                end_pos = pos + int(size)
                dst_ds[pos:end_pos] = src_ds[start_src:end_src]
                pos = end_pos

        # Duplicate the created split into train and val so they contain the same events
        if split in dst:
            for trg_split in ["train", "val"]:
                if trg_split == split:
                    continue
                if trg_split in dst:
                    del dst[trg_split]
                dst.copy(dst[split], trg_split)


def main():
    parser = argparse.ArgumentParser(description="Extract subset of events by ev_id prefix")
    parser.add_argument(
        "--source",
        type=str,
        default="/home2/ivkhar/Baikal/data/normed/baikal_2020_sig-noise_mid-eq_normed.h5",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="/home/plotnikovgp/baikal/data/baikal_2020_sig-noise_mid-eq_normed_nuatm_100k.h5",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--prefix", type=str, default="nuatm")
    parser.add_argument("--num-events", type=int, default=100_000)

    args = parser.parse_args()
    extract_events(
        source_path=args.source,
        target_path="/home/plotnikovgp/baikal/data/baikal_2020_sig-noise_mid-eq_normed_nuatm_100k.h5",
        split=args.split,
        particle_prefix=args.prefix.encode(),
        num_events=args.num_events,
    )


if __name__ == "__main__":
    main()
