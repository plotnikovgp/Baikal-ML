import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def filter_events(
    original_h5_path: str,
    predictions_h5_path: str,
    output_h5_path: str,
    threshold: float = 0.5,
    min_strings: int = 3,
    min_signal_hits: int = 10,
    splits: list = None,
):
    splits = splits or ["train", "val", "test"]

    with h5py.File(original_h5_path, "r") as src, \
         h5py.File(predictions_h5_path, "r") as preds, \
         h5py.File(output_h5_path, "w") as dst:

        for split in splits:
            if f"{split}/ev_starts/data" not in src:
                print(f"Skipping {split}: not found in original file")
                continue
            if f"{split}/data/sig_prob" not in preds:
                print(f"Skipping {split}: not found in predictions file")
                continue

            print(f"\nProcessing {split}...")

            ev_starts_src = src[f"{split}/ev_starts/data"][:]
            pred_ev_starts = preds[f"{split}/ev_starts/data"][:]
            sig_probs = preds[f"{split}/data/sig_prob"][:]

            data_src = src[f"{split}/data/data"]
            labels_src = src[f"{split}/labels/data"] if f"{split}/labels/data" in src else None
            tres_src = src[f"{split}/t_res/data"] if f"{split}/t_res/data" in src else None

            num_events = len(pred_ev_starts) - 1
            selected_indices = []

            for i in tqdm(range(num_events), desc=f"Filtering {split}"):
                pred_start = pred_ev_starts[i]
                pred_end = pred_ev_starts[i + 1]
                probs = sig_probs[pred_start:pred_end]

                signal_mask = probs > threshold
                num_signal_hits = np.sum(signal_mask)

                if num_signal_hits < min_signal_hits:
                    continue

                src_start = ev_starts_src[i]
                src_end = ev_starts_src[i + 1]
                event_data = data_src[src_start:src_end]

                channels = event_data[:, 4].astype(int)
                strings = channels // 36
                signal_strings = strings[signal_mask]
                num_unique_strings = len(np.unique(signal_strings))

                if num_unique_strings >= min_strings:
                    selected_indices.append(i)

            print(f"{split}: {len(selected_indices)}/{num_events} events passed filter")

            if len(selected_indices) == 0:
                print(f"Warning: No events passed filter for {split}")
                continue

            new_data = []
            new_labels = []
            new_tres = []
            new_ev_starts = [0]

            for idx in tqdm(selected_indices, desc=f"Copying {split}"):
                src_start = ev_starts_src[idx]
                src_end = ev_starts_src[idx + 1]

                event_data = data_src[src_start:src_end]
                new_data.append(event_data)
                new_ev_starts.append(new_ev_starts[-1] + len(event_data))

                if labels_src is not None:
                    new_labels.append(labels_src[src_start:src_end])

                if tres_src is not None:
                    new_tres.append(tres_src[src_start:src_end])

            new_data = np.concatenate(new_data, axis=0)
            new_ev_starts = np.array(new_ev_starts, dtype=np.int64)

            dst.create_dataset(f"{split}/data/data", data=new_data)
            dst.create_dataset(f"{split}/ev_starts/data", data=new_ev_starts)

            if new_labels:
                new_labels = np.concatenate(new_labels, axis=0)
                dst.create_dataset(f"{split}/labels/data", data=new_labels)

            if new_tres:
                new_tres = np.concatenate(new_tres, axis=0)
                dst.create_dataset(f"{split}/t_res/data", data=new_tres)

            for key in src[split].keys():
                if key in ["data", "ev_starts", "labels", "t_res"]:
                    continue

                if f"{split}/{key}/data" in src:
                    src_dataset = src[f"{split}/{key}/data"]
                    if len(src_dataset.shape) == 1 and src_dataset.shape[0] == len(ev_starts_src) - 1:
                        new_arr = src_dataset[:][selected_indices]
                        dst.create_dataset(f"{split}/{key}/data", data=new_arr)
                    elif len(src_dataset.shape) == 2 and src_dataset.shape[0] == len(ev_starts_src) - 1:
                        new_arr = src_dataset[:][selected_indices]
                        dst.create_dataset(f"{split}/{key}/data", data=new_arr)

            print(f"{split}: {len(new_data)} hits, {len(selected_indices)} events written")

        if "norm_param" in src:
            src.copy("norm_param", dst)


def main():
    parser = argparse.ArgumentParser(description="Filter H5 events by signal predictions")
    parser.add_argument("-i", "--original", type=str, required=True, help="Path to original H5 file")
    parser.add_argument("-p", "--predictions", type=str, required=True, help="Path to predictions H5 file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output H5 path")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Signal probability threshold")
    parser.add_argument("-ms", "--min-strings", type=int, default=3, help="Minimum unique signal strings")
    parser.add_argument("-mh", "--min-signal-hits", type=int, default=10, help="Minimum signal hits")
    parser.add_argument("-s", "--splits", type=str, nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    original_path = Path(args.original)
    output_path = args.output or Path(f"{original_path.stem}_filtered_t{args.threshold}_s{args.min_strings}_h{args.min_signal_hits}.h5")

    print(f"Original: {args.original}")
    print(f"Predictions: {args.predictions}")
    print(f"Output: {output_path}")
    print(f"Threshold: {args.threshold}")
    print(f"Min strings: {args.min_strings}")
    print(f"Min signal hits: {args.min_signal_hits}")

    filter_events(
        original_h5_path=args.original,
        predictions_h5_path=args.predictions,
        output_h5_path=output_path,
        threshold=args.threshold,
        min_strings=args.min_strings,
        min_signal_hits=args.min_signal_hits,
        splits=args.splits,
    )

    print(f"\nFiltered data saved to {output_path}")


if __name__ == "__main__":
    main()

