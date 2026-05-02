"""
Generate a predictions H5 file with random noise/signal probabilities.
Output format matches predict_to_h5.py so it can be used with filter_by_signal.py.

Usage:
    python create_random_preds.py \
        --data /path/to/normed.h5 \
        --output /path/to/random_preds.h5 \
        --train-events 3000000 --val-events 100000 --test-events 100000 \
        --noise-fraction 0.9
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Generate random noise/signal prediction probabilities"
    )
    parser.add_argument("--data", "-d", type=str, required=True, help="Source normed H5 file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output H5 path")
    parser.add_argument("--train-events", type=int, default=3_000_000)
    parser.add_argument("--val-events", type=int, default=100_000)
    parser.add_argument("--test-events", type=int, default=100_000)
    parser.add_argument(
        "--noise-fraction",
        type=float,
        default=0.9,
        help="Fraction of hits to label as noise (default: 0.9)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    data_path = Path(args.data)
    output_path = args.output or str(data_path.parent / f"{data_path.stem}_random_preds.h5")

    split_limits = {
        "train": args.train_events,
        "val": args.val_events,
        "test": args.test_events,
    }

    print(f"Source: {args.data}")
    print(f"Output: {output_path}")
    print(f"Noise fraction: {args.noise_fraction}")
    print(f"Seed: {args.seed}")

    with h5py.File(args.data, "r") as src, h5py.File(output_path, "w") as dst:
        for split, limit in split_limits.items():
            key = f"{split}/ev_starts/data"
            if key not in src:
                print(f"\nSkipping {split}: not found in source")
                continue

            ev_starts = src[key][:]
            total_events = len(ev_starts) - 1
            n_events = min(limit, total_events)

            sub_ev_starts = ev_starts[: n_events + 1]
            hit_start = int(sub_ev_starts[0])
            hit_end = int(sub_ev_starts[-1])
            total_hits = hit_end - hit_start

            new_ev_starts = sub_ev_starts - sub_ev_starts[0]

            is_noise = rng.random(total_hits) < args.noise_fraction
            sig_prob = np.where(
                is_noise,
                rng.uniform(0.0, 0.3, total_hits),
                rng.uniform(0.7, 1.0, total_hits),
            ).astype(np.float32)

            dst.create_dataset(f"{split}/data/sig_prob", data=sig_prob)
            dst.create_dataset(f"{split}/ev_starts/data", data=new_ev_starts)

            n_noise = int(is_noise.sum())
            print(
                f"\n{split}: {n_events}/{total_events} events, "
                f"{total_hits} hits, {n_noise} noise ({n_noise / total_hits * 100:.1f}%)"
            )

    print(f"\nDone. Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
