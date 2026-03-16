"""
Generate noise-signal predictions for raw (flat) H5 data format.
Output format is compatible with filter_data_multiprocessing.py's take_NN_filt_hits option.

Raw data structure:
    {particle}/{prefix}/data/{part}/data - hit features [N_hits, 5] (Q, time, x, y, z)
    {particle}/{prefix}/ev_starts/{part}/data - event boundaries

Output structure:
    {folder}/{particle}/{part}/nn_noise_hit_prob - predictions [N_hits, 1]
"""

import argparse
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_size,
        num_layers,
        dim_feedforward_size,
        n_heads,
        out_size,
        dropout_p=0.0,
        **kwargs,
    ):
        super().__init__()
        self.first_layer = nn.Linear(in_features, hidden_size)
        enc_layer = nn.TransformerEncoderLayer(
            hidden_size, n_heads, dim_feedforward_size, dropout_p, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers)
        self.head = nn.Linear(hidden_size, out_size, bias=False)

    def forward(self, x, mask):
        mask = (~mask).float()
        x = self.first_layer(x)
        x = self.enc(x, src_key_padding_mask=mask)
        return self.head(x)


def load_model(checkpoint_path, config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = TransformerEncoder(**config["model_params"])
    state_dict = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(state_dict)
    model = model.to("cuda")
    model.eval()

    return model, config


def load_norm_params(norm_h5_path):
    with h5py.File(norm_h5_path, "r") as f:
        mean = np.array(f["norm_param/mean"])
        std = np.array(f["norm_param/std"])
    return mean, std


def normalize_data(data, mean, std):
    return (data - mean) / std


def parse_kv_ints(items):
    if items is None:
        return {}
    limits = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Expected key=value for --limit-num-parts, got: {item}"
            )
        key, value = item.split("=", 1)
        try:
            limits[key] = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Expected integer value for --limit-num-parts, got: {item}"
            ) from exc
    return limits


def collate_batch(ev_starts, raw_data):
    local_starts = ev_starts - ev_starts[0]
    batch_size = len(local_starts) - 1
    max_len = int(np.max(np.diff(local_starts)))
    n_features = raw_data.shape[-1]

    x = np.zeros((batch_size, max_len, n_features), dtype=np.float32)
    mask = np.zeros((batch_size, max_len), dtype=bool)

    for i in range(batch_size):
        start, end = int(local_starts[i]), int(local_starts[i + 1])
        length = end - start
        x[i, :length] = raw_data[start:end]
        mask[i, :length] = True

    return torch.tensor(x), torch.tensor(mask)


def process_particle_part(
    model,
    raw_h5_path,
    particle,
    part,
    prefix,
    mean,
    std,
    batch_size,
    events_limit=None,
):
    all_probs = []

    with h5py.File(raw_h5_path, "r") as f:
        data_path = f"{particle}/{prefix}/data/{part}/data"
        ev_starts_path = f"{particle}/{prefix}/ev_starts/{part}/data"

        if data_path not in f:
            print(f"  Skipping {particle}/{part}: data not found")
            return None

        ev_starts = f[ev_starts_path][:]
        data_dataset = f[data_path]
        num_events = len(ev_starts) - 1

        if events_limit is not None:
            num_events = min(num_events, events_limit)

        num_batches = (num_events + batch_size - 1) // batch_size

        with torch.no_grad():
            for batch_idx in tqdm(
                range(num_batches),
                desc=f"  {particle}/{part}",
                leave=False,
            ):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, num_events)
                batch_ev_starts = ev_starts[batch_start : batch_end + 1]

                global_start = int(batch_ev_starts[0])
                global_end = int(batch_ev_starts[-1])
                batch_data = np.array(data_dataset[global_start:global_end], dtype=np.float32)

                batch_data_norm = normalize_data(batch_data, mean, std)

                x, mask = collate_batch(batch_ev_starts, batch_data_norm)
                x = x.to("cuda")
                mask = mask.to("cuda")

                output = model(x, mask)

                if output.shape[-1] == 2:
                    probs = torch.sigmoid(output[:, :, 1])
                else:
                    probs = torch.sigmoid(output).squeeze(-1)

                probs = probs.cpu().numpy()
                mask_np = mask.cpu().numpy()

                for i in range(len(mask_np)):
                    seq_len = mask_np[i].sum()
                    probs_i = probs[i][:seq_len]
                    all_probs.append(probs_i)

    return np.concatenate(all_probs) if all_probs else None


def main():
    parser = argparse.ArgumentParser(
        description="Generate noise-signal predictions for raw H5 data"
    )
    parser.add_argument(
        "--checkpoint", "-ck", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to model config yaml")
    parser.add_argument(
        "--raw-data", "-r", type=str, required=True, help="Path to raw (flat) H5 file"
    )
    parser.add_argument(
        "--norm-data",
        "-n",
        type=str,
        required=True,
        help="Path to H5 file with norm_param (trained data)",
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Output H5 path")
    parser.add_argument(
        "--output-folder",
        "-f",
        type=str,
        default="preds",
        help="Folder name in output H5 (default: preds)",
    )
    parser.add_argument(
        "--particles",
        "-p",
        type=str,
        nargs="+",
        default=["muatm", "nuatm", "nue2"],
        help="Particles to process",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="raw",
        help="Data prefix in raw H5 (raw or reco)",
    )
    parser.add_argument("--batch-size", "-b", type=int, default=256)
    parser.add_argument(
        "--events-limit", "-l", type=int, default=None, help="Limit events per part"
    )
    parser.add_argument(
        "--limit-num-parts",
        type=str,
        nargs="+",
        default=None,
        help="Limit number of parts per particle: muatm=400 nuatm=260",
    )
    parser.add_argument(
        "--sample-parts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample parts instead of taking first N (default: True)",
    )
    parser.add_argument(
        "--nn-threshold",
        type=float,
        default=None,
        help="Store NN threshold in metadata (not used here)",
    )
    args = parser.parse_args()
    limit_num_parts = parse_kv_ints(args.limit_num_parts)

    print(f"Loading model from {args.checkpoint}")
    model, config = load_model(args.checkpoint, args.config)

    print(f"Loading normalization params from {args.norm_data}")
    mean, std = load_norm_params(args.norm_data)
    print(f"  Mean: {mean}")
    print(f"  Std: {std}")

    raw_path = Path(args.raw_data)
    output_path = args.output or f"{raw_path.stem}_nn_preds.h5"

    print(f"\nProcessing {args.raw_data}")
    print(f"Output: {output_path}")
    print(f"Output folder: {args.output_folder}")

    with h5py.File(args.raw_data, "r") as src, h5py.File(output_path, "w") as dst:
        meta = dst.create_group("meta")
        meta.attrs["checkpoint_path"] = str(args.checkpoint)
        meta.attrs["config_path"] = str(args.config)
        meta.attrs["raw_data"] = str(args.raw_data)
        meta.attrs["norm_data"] = str(args.norm_data)
        meta.attrs["output_folder"] = str(args.output_folder)
        meta.attrs["prefix"] = str(args.prefix)
        meta.attrs["particles"] = ",".join(args.particles)
        meta.attrs["batch_size"] = int(args.batch_size)
        meta.attrs["events_limit"] = int(args.events_limit) if args.events_limit is not None else -1
        meta.attrs["nn_threshold"] = (
            float(args.nn_threshold) if args.nn_threshold is not None else -1.0
        )
        meta.attrs["limit_num_parts"] = ",".join(f"{k}={v}" for k, v in limit_num_parts.items())
        with open(args.config, "r") as f:
            meta.create_dataset(
                "config_yaml",
                data=np.string_(f.read()),
            )
        for particle in args.particles:
            if particle not in src:
                print(f"\nSkipping {particle}: not found in file")
                continue

            print(f"\nProcessing {particle}...")

            data_key = f"{particle}/{args.prefix}/data"
            if data_key not in src:
                print(f"  Skipping: {data_key} not found")
                continue

            parts = list(src[data_key].keys())
            limit_parts = limit_num_parts.get(particle)
            if limit_parts is not None and limit_parts > 0:
                if args.sample_parts:
                    parts = random.sample(parts, k=min(limit_parts, len(parts)))
                else:
                    parts = parts[:limit_parts]
            print(f"  Found {len(parts)} parts")

            for part in tqdm(parts, desc=f"  {particle} parts"):
                probs = process_particle_part(
                    model=model,
                    raw_h5_path=args.raw_data,
                    particle=particle,
                    part=part,
                    prefix=args.prefix,
                    mean=mean,
                    std=std,
                    batch_size=args.batch_size,
                    events_limit=args.events_limit,
                )

                if probs is not None:
                    preds_path = f"{args.output_folder}/{particle}/{part}/`preds`"
                    dst.create_dataset(
                        preds_path,
                        data=probs.reshape(-1, 1),
                        dtype=np.float32,
                        compression="gzip",
                    )

    print(f"\nPredictions saved to {output_path}")
    print("Use with filter_data_multiprocessing.py:")
    print(f'  "h5_nn_preds": "{output_path}",')
    print(f'  "h5_nn_preds_folder": "{args.output_folder}",')


if __name__ == "__main__":
    main()
