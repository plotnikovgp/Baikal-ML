"""Compare per-event-type P/R/AUC/AP under two signal definitions:

  A:  |t_res| < 10  OR  label != 0   (current, matches NoiseSigOrLabelsPreprocessor)
  B:  |t_res| < 10  OR  label > 0    (hypothetical "old" def)

Runs the `i_tres10_or_labels_merged_smaller_hs128` model once and prints a diff.
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
from models.encoder import Encoder  # noqa: E402
from scripts.generate_report import denormalize, load_norm_params  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MC_PATH = "/home/plotnikovgp/baikal/data/baikal_mc_merged_m50-na15-nue15_all_smaller_norm.h5"
CKPT = "checkpoints/noise_sig_experiments/i_tres10_or_labels_merged_smaller_hs128/best.ckpt"


def iter_batches(h5_path, split, max_per_type, batch_size=128):
    src_mean, src_std = load_norm_params(h5_path)
    with h5py.File(h5_path, "r") as f:
        ev_starts = f[f"{split}/ev_starts/data"][:]
        data_all = f[f"{split}/data/data"]
        t_res_all = f[f"{split}/t_res/data"]
        labels_all = f[f"{split}/labels/data"]
        ev_ids = f[f"{split}/ev_ids/data"][:]

        type_idx = {}
        for i in range(len(ev_starts) - 1):
            t = ev_ids[i].decode().split("_")[0]
            type_idx.setdefault(t, []).append(i)

        indices = []
        for t, idxs in type_idx.items():
            sel = idxs[:max_per_type]
            indices.extend(sel)
            print(f"  {t}: {len(sel)}")
        indices.sort()

        for bs_start in range(0, len(indices), batch_size):
            batch = indices[bs_start : bs_start + batch_size]
            evs, tres_b, lab_b, ty_b = [], [], [], []
            for idx in batch:
                s, e = int(ev_starts[idx]), int(ev_starts[idx + 1])
                hits = data_all[s:e].astype(np.float32)
                evs.append(hits)
                tres_b.append(t_res_all[s:e])
                lab_b.append(labels_all[s:e])
                ty_b.append(ev_ids[idx].decode().split("_")[0])

            max_len = max(len(ev) for ev in evs)
            bs = len(evs)
            x = np.zeros((bs, max_len, 5), dtype=np.float32)
            m = np.zeros((bs, max_len), dtype=np.float32)
            tres = np.zeros((bs, max_len), dtype=np.float32)
            lab = np.zeros((bs, max_len), dtype=np.float32)
            for i, ev in enumerate(evs):
                L = len(ev)
                x[i, :L] = ev
                m[i, :L] = 1.0
                tres[i, :L] = tres_b[i]
                lab[i, :L] = lab_b[i]
            yield torch.tensor(x), torch.tensor(m), tres, lab, ty_b


def run(args):
    model = Encoder(
        in_features=5,
        hidden_size=128,
        num_layers=5,
        dim_feedforward_size=512,
        n_heads=1,
        out_size=2,
        dropout_p=0.0,
    )
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model = model.to(DEVICE).eval()

    print(f"Loading events from {args.split} (max {args.max_per_type} per type)...")
    batches = list(iter_batches(args.mc_path, args.split, args.max_per_type))

    # Collect per-hit probs, tres, labels, and per-hit ev-type tag
    probs_by_type = {"muatm": [], "nuatm": [], "nue2": []}
    tres_by_type = {k: [] for k in probs_by_type}
    lab_by_type = {k: [] for k in probs_by_type}

    with torch.no_grad():
        for x, mask, tres, lab, tys in tqdm(batches, desc="Inference"):
            out = model(x.to(DEVICE), mask.to(DEVICE).bool())
            if isinstance(out, tuple):
                out = out[0]
            p = torch.sigmoid(out[:, :, 1]).cpu().numpy()
            m = mask.numpy().astype(bool)
            for i, t in enumerate(tys):
                mi = m[i]
                if t not in probs_by_type:
                    continue
                probs_by_type[t].append(p[i][mi])
                tres_by_type[t].append(tres[i][mi])
                lab_by_type[t].append(lab[i][mi])

    thr = args.threshold

    def pr_at(probs, y):
        pred = probs > thr
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) else 1.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        return p, r

    def metrics(probs, y):
        if len(np.unique(y)) < 2:
            return None, None, *pr_at(probs, y)
        return roc_auc_score(y, probs), average_precision_score(y, probs), *pr_at(probs, y)

    def fmt(v):
        return "  n/a " if v is None else f"{v:.4f}"

    print("\nPer-event-type metrics (threshold = {:.2f})".format(thr))
    print(
        f"{'type':<8} {'n_hits':>10} "
        f"{'AUC_!=0':>9} {'AUC_>0':>9} {'Δ':>8}  "
        f"{'AP_!=0':>9} {'AP_>0':>9} {'Δ':>8}  "
        f"{'P_!=0':>8} {'P_>0':>8} {'Δ':>8}  "
        f"{'R_!=0':>8} {'R_>0':>8} {'Δ':>8}"
    )
    print("-" * 160)

    all_probs_A, all_y_A, all_probs_B, all_y_B = [], [], [], []
    for t in ("muatm", "nuatm", "nue2"):
        if not probs_by_type[t]:
            continue
        probs = np.concatenate(probs_by_type[t])
        tres = np.concatenate(tres_by_type[t])
        lab = np.concatenate(lab_by_type[t])
        y_A = ((np.abs(tres) < 10) | (lab != 0)).astype(np.int8)
        y_B = ((np.abs(tres) < 10) | (lab > 0)).astype(np.int8)
        auc_A, ap_A, p_A, r_A = metrics(probs, y_A)
        auc_B, ap_B, p_B, r_B = metrics(probs, y_B)

        print(
            f"{t:<8} {len(probs):>10,} "
            f"{fmt(auc_A):>9} {fmt(auc_B):>9} {auc_A - auc_B:+.4f}  "
            f"{fmt(ap_A):>9} {fmt(ap_B):>9} {ap_A - ap_B:+.4f}  "
            f"{fmt(p_A):>8} {fmt(p_B):>8} {p_A - p_B:+.4f}  "
            f"{fmt(r_A):>8} {fmt(r_B):>8} {r_A - r_B:+.4f}"
        )
        all_probs_A.append(probs)
        all_y_A.append(y_A)
        all_probs_B.append(probs)
        all_y_B.append(y_B)

    p_all = np.concatenate(all_probs_A)
    yA_all = np.concatenate(all_y_A)
    yB_all = np.concatenate(all_y_B)
    auc_A, ap_A, p_A, r_A = metrics(p_all, yA_all)
    auc_B, ap_B, p_B, r_B = metrics(p_all, yB_all)
    print("-" * 160)
    print(
        f"{'OVERALL':<8} {len(p_all):>10,} "
        f"{fmt(auc_A):>9} {fmt(auc_B):>9} {auc_A - auc_B:+.4f}  "
        f"{fmt(ap_A):>9} {fmt(ap_B):>9} {ap_A - ap_B:+.4f}  "
        f"{fmt(p_A):>8} {fmt(p_B):>8} {p_A - p_B:+.4f}  "
        f"{fmt(r_A):>8} {fmt(r_B):>8} {r_A - r_B:+.4f}"
    )

    n_neg = int(((yA_all == 1) & (yB_all == 0)).sum())
    print(
        f"\nHits flipped (signal under !=0 but not under >0): "
        f"{n_neg:,} / {len(p_all):,} = {n_neg / len(p_all) * 100:.4f}%"
    )
    _ = denormalize  # silence unused import


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--mc-path", default=MC_PATH)
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-per-type", type=int, default=10_000)
    ap.add_argument("--threshold", type=float, default=0.5)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
