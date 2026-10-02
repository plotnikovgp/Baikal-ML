"""Write masked per-hit pseudo-labels for EXP data.

Labels:
    1  if P(signal) > p_hi   (confident signal)
    0  if P(signal) < p_lo   (confident noise)
   -1  otherwise             (ambiguous, masked from loss via ignore_index)

Sidecar h5 layout (per split):
    {split}/ev_starts/data    int64
    {split}/labels/data       int32   {-1, 0, 1}

Usage:
    python scripts/iter_pseudolabel/make_masked_pseudolabels.py \
        --ckpt checkpoints/.../best.ckpt \
        --p-lo 0.1 --p-hi 0.9 \
        --out data/pseudolabels/masked_0p1_0p9/iter_1.h5
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
    ap.add_argument("--p-lo", type=float, required=True, help="Below this → label 0 (noise)")
    ap.add_argument("--p-hi", type=float, required=True, help="Above this → label 1 (signal)")
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
        "p_lo": float(args.p_lo),
        "p_hi": float(args.p_hi),
        "exp_h5": str(args.exp_h5),
        "out": str(out_path),
        "splits": {},
    }

    with h5py.File(out_path, "w") as out_h5:
        for split in args.splits:
            print(f"[masked-pl {args.p_lo}/{args.p_hi}] split={split}")
            probs = infer_split_probs(
                model, args.exp_h5, split, src_mean, src_std, dst_mean, dst_std, args.batch_size
            )
            labels = np.full(probs.shape, -1, dtype=np.int32)
            labels[probs < args.p_lo] = 0
            labels[probs > args.p_hi] = 1

            with h5py.File(args.exp_h5, "r") as src:
                ev_starts = src[f"{split}/ev_starts/data"][:]
            out_h5.create_dataset(f"{split}/ev_starts/data", data=ev_starts)
            out_h5.create_dataset(f"{split}/labels/data", data=labels)

            n = labels.size
            stats = {
                "n_hits": int(n),
                "n_events": int(len(ev_starts) - 1),
                "frac_noise": float((labels == 0).sum() / n) if n else None,
                "frac_signal": float((labels == 1).sum() / n) if n else None,
                "frac_masked": float((labels == -1).sum() / n) if n else None,
                "mean_prob": float(probs.mean()) if n else None,
            }
            print(
                f"  hits={stats['n_hits']}, noise={stats['frac_noise']:.4f}, "
                f"signal={stats['frac_signal']:.4f}, masked={stats['frac_masked']:.4f}"
            )
            report["splits"][split] = stats

    out_path.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"[masked-pl] Wrote {out_path} and {out_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
