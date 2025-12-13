import argparse
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


def collate_batch(ev_starts, raw_data):
    local_starts = ev_starts - ev_starts[0]
    batch_size = len(local_starts) - 1
    max_len = int(np.max(np.diff(local_starts)))
    n_features = raw_data.shape[-1]

    x = np.zeros((batch_size, max_len, n_features), dtype=np.float32)
    mask = np.zeros((batch_size, max_len), dtype=bool)

    for i in range(batch_size):
        start, end = local_starts[i], local_starts[i + 1]
        length = end - start
        x[i, :length] = raw_data[start:end]
        mask[i, :length] = True

    return torch.tensor(x), torch.tensor(mask)


def batch_generator(h5_path, split, batch_size, events_limit=None):
    with h5py.File(h5_path, "r") as f:
        ev_starts = f[f"{split}/ev_starts/data"][:]
        data_dataset = f[f"{split}/data/data"]
        num_events = len(ev_starts) - 1

        if events_limit is not None:
            num_events = min(num_events, events_limit)

        for batch_start in range(0, num_events, batch_size):
            batch_end = min(batch_start + batch_size, num_events)
            batch_ev_starts = ev_starts[batch_start : batch_end + 1]

            global_start = int(batch_ev_starts[0])
            global_end = int(batch_ev_starts[-1])
            batch_data = data_dataset[global_start:global_end]

            x, mask = collate_batch(batch_ev_starts, batch_data)
            yield x, mask


def count_batches(h5_path, split, batch_size, events_limit=None):
    with h5py.File(h5_path, "r") as f:
        num_events = len(f[f"{split}/ev_starts/data"]) - 1
        if events_limit is not None:
            num_events = min(num_events, events_limit)
        num_batches = (num_events + batch_size - 1) // batch_size
        print(f"{split}: {num_events} events, {num_batches} batches")
    return num_batches


def get_predictions_for_split(model, h5_path, split, batch_size, events_limit=None):
    all_probs = []
    total_hits = 0
    total_batches = count_batches(h5_path, split, batch_size, events_limit)

    with torch.no_grad():
        for x, mask in tqdm(
            batch_generator(h5_path, split, batch_size, events_limit),
            desc=f"Processing {split}",
            total=total_batches,
        ):
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
                probs_i = probs[i][mask_np[i]]
                all_probs.append(probs_i)
                total_hits += len(probs_i)

    print(f"{split}: {len(all_probs)} events, {total_hits} hits")
    return all_probs


def write_predictions_to_h5(input_h5_path, output_h5_path, predictions_dict):
    with h5py.File(input_h5_path, "r") as src, h5py.File(output_h5_path, "w") as dst:
        for split_name in ["train", "val", "test"]:
            if split_name not in predictions_dict:
                continue

            probs_list = predictions_dict[split_name]
            probs_flat = np.concatenate(probs_list).astype(np.float32)
            num_events = len(probs_list)

            dst.create_dataset(f"{split_name}/data/sig_prob", data=probs_flat)

            if f"{split_name}/ev_starts/data" in src:
                ev_starts = src[f"{split_name}/ev_starts/data"][: num_events + 1]
                recalc_ev_starts = np.zeros(num_events + 1, dtype=ev_starts.dtype)
                recalc_ev_starts[0] = 0
                for i, probs in enumerate(probs_list):
                    recalc_ev_starts[i + 1] = recalc_ev_starts[i] + len(probs)
                dst.create_dataset(f"{split_name}/ev_starts/data", data=recalc_ev_starts)

            print(f"{split_name}: {len(probs_flat)} predictions, {num_events} events written")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ck", "--checkpoint",
        type=str,
        default="/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/checkpoints/noise_sig_rerun_v2/encoder_nl5_nh1_dff512_hs512_bs128/best_2020.ckpt",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/train_configs/noise_sig.yaml",
    )
    parser.add_argument(
        "-d", "--data",
        type=str,
        default="/home2/ivkhar/Baikal/data/normed/baikal_2020_sig-noise_mid-eq_normed.h5",
    )
    parser.add_argument("-v", "--version", type=int, default=1)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-s", "--splits", type=str, nargs="+", default=["train", "val", "test"])
    parser.add_argument("-o", "--output-path", type=str, default=None)
    parser.add_argument("-l", "--events-limit", type=int, default=None, help="Limit the number of events to process")
    args = parser.parse_args()

    model, config = load_model(args.checkpoint, args.config)

    data_path = Path(args.data)
    output_path = args.output_path or Path(f"{data_path.stem}_nn_v{args.version}_sig_probs.h5")

    predictions = {}

    for split in args.splits:
        print(f"\nProcessing {split} split...")
        predictions[split] = get_predictions_for_split(model, args.data, split, args.batch_size, args.events_limit)

    write_predictions_to_h5(args.data, output_path, predictions)
    print(f"\nPredictions saved to {output_path}")


if __name__ == "__main__":
    main()
