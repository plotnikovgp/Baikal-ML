"""Copy contiguous event prefixes of experimental splits into a small HDF5."""

from __future__ import annotations

import argparse

import h5py


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--train", type=int, default=60000)
    parser.add_argument("--val", type=int, default=10000)
    parser.add_argument("--test", type=int, default=10000)
    args = parser.parse_args()

    with h5py.File(args.source, "r") as src, h5py.File(args.output, "w") as dst:
        src.copy("norm_param", dst)
        for split, n_requested in (
            ("train", args.train),
            ("val", args.val),
            ("test", args.test),
        ):
            group = src[split]
            starts = group["ev_starts/data"]
            n = min(n_requested, len(starts) - 1)
            n_hits = int(starts[n])
            dst_group = dst.create_group(split)
            for name in group:
                src_data = group[f"{name}/data"]
                out_group = dst_group.create_group(name)
                size = (
                    n + 1
                    if name == "ev_starts"
                    else n_hits
                    if name in {"data", "channels"}
                    else n
                )
                out_group.create_dataset("data", data=src_data[:size])
            print(split, n, n_hits, flush=True)


if __name__ == "__main__":
    main()
