"""Evaluate a DA (domain-adaptation) noise/signal checkpoint.

Computes per-event-type AUC / AP / P@thr / R@thr on MC data using the model's
training signal definition (label != 0 — `NoiseSigOriginalLabelsPreprocessor`).

Default checkpoint: k_nsol_labelneq0_da_hs128_k0p001 / best_mc_2020.ckpt.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.encoder import EncoderDomainAdaptation  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_CKPT = (
    "checkpoints/noise_sig_experiments/k_nsol_labelneq0_da_hs128_k0p001/best_mc_2020.ckpt"
)
DEFAULT_MC = (
    "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_2020_balanced_val_2k_each.h5"
)
DEFAULT_RENORM_TO = "/home2/ivkhar/Baikal/data/normed/baikal_2020_sig-noise_mid-eq_normed.h5"


def load_norm(h5_path):
    with h5py.File(h5_path, "r") as f:
        return (
            f["norm_param/mean"][:].astype(np.float32),
            f["norm_param/std"][:].astype(np.float32),
        )


def iter_batches(h5_path, split, max_per_type, renorm=None, batch_size=128):
    """renorm = (src_mean, src_std, dst_mean, dst_std) or None."""
    with h5py.File(h5_path, "r") as f:
        ev_starts = f[f"{split}/ev_starts/data"][:]
        data_all = f[f"{split}/data/data"]
        labels_all = f[f"{split}/labels/data"]
        ev_ids = f[f"{split}/ev_ids/data"][:]

        type_idx = {}
        for i in range(len(ev_starts) - 1):
            t = ev_ids[i].decode().split("_")[0]
            type_idx.setdefault(t, []).append(i)

        indices = []
        for t, idxs in type_idx.items():
            sel = idxs if max_per_type is None else idxs[:max_per_type]
            indices.extend(sel)
            print(f"  {t}: {len(sel)}")
        indices.sort()

        for bs_start in range(0, len(indices), batch_size):
            batch = indices[bs_start : bs_start + batch_size]
            evs, lab_b, ty_b = [], [], []
            for idx in batch:
                s, e = int(ev_starts[idx]), int(ev_starts[idx + 1])
                hits = data_all[s:e].astype(np.float32)
                if renorm is not None:
                    src_m, src_s, dst_m, dst_s = renorm
                    hits = (hits * src_s + src_m - dst_m) / dst_s
                evs.append(hits)
                lab_b.append(labels_all[s:e])
                ty_b.append(ev_ids[idx].decode().split("_")[0])

            max_len = max(len(ev) for ev in evs)
            bs = len(evs)
            x = np.zeros((bs, max_len, 5), dtype=np.float32)
            m = np.zeros((bs, max_len), dtype=np.float32)
            lab = np.zeros((bs, max_len), dtype=np.float32)
            for i, ev in enumerate(evs):
                L = len(ev)
                x[i, :L] = ev
                m[i, :L] = 1.0
                lab[i, :L] = lab_b[i]
            yield torch.tensor(x), torch.tensor(m), lab, ty_b


def load_da_model(ckpt_path, hs, dff, num_layers, n_heads, num_domains, dc_hs, dc_layers):
    model = EncoderDomainAdaptation(
        in_features=5,
        hidden_size=hs,
        num_layers=num_layers,
        dim_feedforward_size=dff,
        n_heads=n_heads,
        out_size=2,
        dropout_p=0.0,
        num_domains=num_domains,
        domain_classifier_hidden_size=dc_hs,
        domain_classifier_layers=dc_layers,
        gradient_reversal_alpha=1.0,
        aggregate_output=False,
    )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    return model.to(DEVICE).eval()


def pr_at(probs, y, thr):
    pred = probs > thr
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r


def metrics(probs, y, thr):
    if len(np.unique(y)) < 2:
        p, r = pr_at(probs, y, thr)
        return None, None, p, r
    return roc_auc_score(y, probs), average_precision_score(y, probs), *pr_at(probs, y, thr)


def fmt(v):
    return "  n/a " if v is None else f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--mc-path", default=DEFAULT_MC)
    ap.add_argument(
        "--renorm-to",
        default=DEFAULT_RENORM_TO,
        help="Path to the file whose norm-params the model expects. "
        "Pass empty string to disable renormalization.",
    )
    ap.add_argument("--split", default="val")
    ap.add_argument(
        "--max-per-type", type=int, default=None, help="None = all; otherwise cap per type"
    )
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--hs", type=int, default=128)
    ap.add_argument("--dff", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=5)
    ap.add_argument("--n-heads", type=int, default=1)
    ap.add_argument("--num-domains", type=int, default=2)
    ap.add_argument("--domain-classifier-hidden-size", type=int, default=128)
    ap.add_argument("--domain-classifier-layers", type=int, default=2)
    args = ap.parse_args()

    print(f"Loading DA model: {args.checkpoint}")
    print(f"  hs={args.hs} dff={args.dff} nl={args.num_layers} nh={args.n_heads}")
    model = load_da_model(
        args.checkpoint,
        hs=args.hs,
        dff=args.dff,
        num_layers=args.num_layers,
        n_heads=args.n_heads,
        num_domains=args.num_domains,
        dc_hs=args.domain_classifier_hidden_size,
        dc_layers=args.domain_classifier_layers,
    )

    renorm = None
    if args.renorm_to and args.renorm_to != args.mc_path:
        src_m, src_s = load_norm(args.mc_path)
        dst_m, dst_s = load_norm(args.renorm_to)
        renorm = (src_m, src_s, dst_m, dst_s)
        print(f"\nRenormalizing inputs: {Path(args.mc_path).name} -> {Path(args.renorm_to).name}")

    print(f"\nReading MC events from {args.mc_path} ({args.split}):")
    batches = list(iter_batches(args.mc_path, args.split, args.max_per_type, renorm=renorm))

    probs_by_type, lab_by_type = {}, {}
    with torch.no_grad():
        for x, mask, lab, tys in tqdm(batches, desc="Inference"):
            out, _ = model(x.to(DEVICE), mask.to(DEVICE).bool())
            p = torch.sigmoid(out[:, :, 1]).cpu().numpy()
            m = mask.numpy().astype(bool)
            for i, t in enumerate(tys):
                mi = m[i]
                probs_by_type.setdefault(t, []).append(p[i][mi])
                lab_by_type.setdefault(t, []).append(lab[i][mi])

    thr = args.threshold
    print(f"\nSignal definition: label != 0   |   threshold = {thr}")
    print(
        f"{'type':<10} {'n_hits':>12} {'n_sig':>12} {'sig%':>7}  "
        f"{'AUC':>8} {'AP':>8} {'P':>8} {'R':>8}"
    )
    print("-" * 90)

    all_p, all_y = [], []
    for t in sorted(probs_by_type):
        probs = np.concatenate(probs_by_type[t])
        lab = np.concatenate(lab_by_type[t])
        y = (lab != 0).astype(np.int8)
        auc, apr, p, r = metrics(probs, y, thr)
        n_pos = int(y.sum())
        print(
            f"{t:<10} {len(probs):>12,} {n_pos:>12,} {n_pos / len(y) * 100:>6.2f}%  "
            f"{fmt(auc):>8} {fmt(apr):>8} {fmt(p):>8} {fmt(r):>8}"
        )
        all_p.append(probs)
        all_y.append(y)

    print("-" * 90)
    p_all = np.concatenate(all_p)
    y_all = np.concatenate(all_y)
    auc, apr, p, r = metrics(p_all, y_all, thr)
    n_pos = int(y_all.sum())
    print(
        f"{'OVERALL':<10} {len(p_all):>12,} {n_pos:>12,} {n_pos / len(y_all) * 100:>6.2f}%  "
        f"{fmt(auc):>8} {fmt(apr):>8} {fmt(p):>8} {fmt(r):>8}"
    )


if __name__ == "__main__":
    main()
