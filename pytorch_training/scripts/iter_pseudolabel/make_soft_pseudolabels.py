"""Write soft per-hit pseudo-labels (sig_prob ∈ [0, 1]) for EXP data.

Sidecar h5 layout (per split):
    {split}/ev_starts/data    int64   (copied from EXP source)
    {split}/sig_prob/data     float32 one prob per hit (P(signal))

Usage:
    python scripts/iter_pseudolabel/make_soft_pseudolabels.py \
        --ckpt checkpoints/.../best.ckpt \
        --exp-h5 data/baikal_exp_filtered_norm_w_model_enc_v2_fix_q.h5 \
        --mc-norm data/baikal_2020_sig-noise_mid-eq_normed.h5 \
        --out data/pseudolabels/soft/iter_1.h5 \
        --splits train val test
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.encoder import Encoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXP_DEFAULT = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_exp_filtered_norm_w_model_enc_v2_fix_q.h5"
MC_NORM_DEFAULT = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_2020_sig-noise_mid-eq_normed.h5"


def load_norm(h5_path):
    with h5py.File(h5_path, "r") as f:
        return f["norm_param/mean"][:].astype(np.float32), f["norm_param/std"][:].astype(np.float32)


def load_encoder(ckpt_path, hidden_size=128, num_layers=5, dim_feedforward=512):
    model = Encoder(
        in_features=5,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dim_feedforward_size=dim_feedforward,
        n_heads=1,
        out_size=2,
        dropout_p=0.0,
    )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    return model.to(DEVICE).eval()


@torch.no_grad()
def infer_split_probs(model, exp_h5, split, src_mean, src_std, dst_mean, dst_std, batch_size):
    """Return a single flat float32 array of P(signal) aligned with EXP `{split}/data/data`."""
    with h5py.File(exp_h5, "r") as f:
        ev_starts = f[f"{split}/ev_starts/data"][:]
        data_all = f[f"{split}/data/data"]
        n_events = len(ev_starts) - 1
        total_hits = int(ev_starts[-1])

        out = np.zeros(total_hits, dtype=np.float32)
        for bs_start in tqdm(
            range(0, n_events, batch_size),
            desc=f"soft pseudo-labels {split}",
            unit="batch",
        ):
            bs_end = min(bs_start + batch_size, n_events)
            ev_slice = ev_starts[bs_start : bs_end + 1]
            global_start = int(ev_slice[0])
            global_end = int(ev_slice[-1])

            hits_block = data_all[global_start:global_end].astype(np.float32)
            hits_block = (hits_block * src_std + src_mean - dst_mean) / dst_std

            seq_lens = (ev_slice[1:] - ev_slice[:-1]).astype(np.int64)
            max_len = int(seq_lens.max())
            bs = bs_end - bs_start
            x = np.zeros((bs, max_len, 5), dtype=np.float32)
            mask = np.zeros((bs, max_len), dtype=bool)
            for i in range(bs):
                L = int(seq_lens[i])
                local_s = int(ev_slice[i] - global_start)
                x[i, :L] = hits_block[local_s : local_s + L]
                mask[i, :L] = True

            x_t = torch.from_numpy(x).to(DEVICE)
            mask_t = torch.from_numpy(mask).to(DEVICE)
            logits = model(x_t, mask_t)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.sigmoid(logits[:, :, 1] - logits[:, :, 0]).cpu().numpy()

            for i in range(bs):
                L = int(seq_lens[i])
                local_s = int(ev_slice[i] - global_start)
                out[global_start + local_s : global_start + local_s + L] = probs[i, :L]

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--exp-h5", default=EXP_DEFAULT)
    ap.add_argument(
        "--mc-norm",
        default=MC_NORM_DEFAULT,
        help="MC h5 whose norm_param defines the model's input space (renorm target)",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden-size", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=5)
    ap.add_argument("--dim-feedforward", type=int, default=512)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_encoder(args.ckpt, args.hidden_size, args.num_layers, args.dim_feedforward)

    src_mean, src_std = load_norm(args.exp_h5)
    dst_mean, dst_std = load_norm(args.mc_norm)

    report = {
        "ckpt": str(args.ckpt),
        "exp_h5": str(args.exp_h5),
        "mc_norm": str(args.mc_norm),
        "out": str(out_path),
        "splits": {},
    }

    with h5py.File(out_path, "w") as out_h5:
        for split in args.splits:
            print(f"[soft-pl] split={split}")
            probs = infer_split_probs(
                model, args.exp_h5, split, src_mean, src_std, dst_mean, dst_std, args.batch_size
            )
            with h5py.File(args.exp_h5, "r") as src:
                ev_starts = src[f"{split}/ev_starts/data"][:]
            out_h5.create_dataset(f"{split}/ev_starts/data", data=ev_starts)
            out_h5.create_dataset(f"{split}/sig_prob/data", data=probs.astype(np.float32))
            stats = {
                "n_hits": int(probs.size),
                "n_events": int(len(ev_starts) - 1),
                "mean_prob": float(probs.mean()) if probs.size else None,
                "frac_gt_0_5": float((probs > 0.5).mean()) if probs.size else None,
                "frac_gt_0_75": float((probs > 0.75).mean()) if probs.size else None,
                "frac_gt_0_9": float((probs > 0.9).mean()) if probs.size else None,
            }
            print(
                f"  hits={stats['n_hits']}, events={stats['n_events']}, "
                f"mean_p={stats['mean_prob']:.4f}, frac>0.5={stats['frac_gt_0_5']:.4f}, "
                f"frac>0.9={stats['frac_gt_0_9']:.4f}"
            )
            report["splits"][split] = stats

    (out_path.with_suffix(".json")).write_text(json.dumps(report, indent=2))
    print(f"[soft-pl] Wrote {out_path} and {out_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
