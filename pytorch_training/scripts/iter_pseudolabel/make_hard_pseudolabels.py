"""Write hard per-hit pseudo-labels (0/1) for EXP data at a fixed threshold.

Sidecar h5 layout (per split):
    {split}/ev_starts/data    int64   (copied from EXP source)
    {split}/labels/data       int32   1 if P(signal) > threshold else 0

Usage:
    python scripts/iter_pseudolabel/make_hard_pseudolabels.py \
        --ckpt checkpoints/.../best.ckpt \
        --threshold 0.75 \
        --exp-h5 data/baikal_exp_filtered_norm_w_model_enc_v2_fix_q.h5 \
        --mc-norm data/baikal_2020_sig-noise_mid-eq_normed.h5 \
        --out data/pseudolabels/hard_t0_75/iter_1.h5 \
        --splits train val test
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_soft_pseudolabels import (
    EXP_DEFAULT,
    MC_NORM_DEFAULT,
    infer_split_probs,
    load_encoder,
    load_norm,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--exp-h5", default=EXP_DEFAULT)
    ap.add_argument("--mc-norm", default=MC_NORM_DEFAULT)
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
        "threshold": float(args.threshold),
        "exp_h5": str(args.exp_h5),
        "mc_norm": str(args.mc_norm),
        "out": str(out_path),
        "splits": {},
    }

    with h5py.File(out_path, "w") as out_h5:
        for split in args.splits:
            print(f"[hard-pl t={args.threshold}] split={split}")
            probs = infer_split_probs(
                model, args.exp_h5, split, src_mean, src_std, dst_mean, dst_std, args.batch_size
            )
            labels = (probs > args.threshold).astype(np.int32)
            with h5py.File(args.exp_h5, "r") as src:
                ev_starts = src[f"{split}/ev_starts/data"][:]
            out_h5.create_dataset(f"{split}/ev_starts/data", data=ev_starts)
            out_h5.create_dataset(f"{split}/labels/data", data=labels)
            stats = {
                "n_hits": int(labels.size),
                "n_events": int(len(ev_starts) - 1),
                "frac_signal": float(labels.mean()) if labels.size else None,
                "mean_prob": float(probs.mean()) if probs.size else None,
            }
            print(
                f"  hits={stats['n_hits']}, events={stats['n_events']}, "
                f"frac_signal={stats['frac_signal']:.4f}, mean_p={stats['mean_prob']:.4f}"
            )
            report["splits"][split] = stats

    out_path.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"[hard-pl] Wrote {out_path} and {out_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
