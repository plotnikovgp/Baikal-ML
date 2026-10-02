"""MC EAS vs EXP raw-count histograms of hit time and predicted |t_res|.

Same style/selection as plots/baseline_vs_hard/baseline.png (raw counts, equal
number of events after the 8-2 cut, rows = classification thresholds), but the
signal hits feeding the histograms also pass a charge cut Q > q_min (default 1).

Columns: (0) raw hit time [ns], (1) predicted |t_res| [ns].
Rows:    one per classification threshold (default 0.5 and 0.8).

t_res requires the two-head model (noise_sig_tres_abs_merged). Its checkpoint
uses an MLP t_res head (128->256->1, GELU); we build that head explicitly so
loading is robust regardless of the encoder.py revision in the tree.
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.encoder import EncoderTwoHead

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MC_COL, EXP_COL = "#2ca02c", "#1f77b4"
FEAT_Q, FEAT_TIME = 0, 1  # data feature order: [Q, T, X, Y, Z]

MC_DEFAULT = "/home/plotnikovgp/baikal/data/baikal_mc_merged_m50-na15-nue15_all_smaller_norm.h5"
EXP_DEFAULT = "data/baikal_exp_filtered_norm_w_model_enc_v2_fix_q.h5"
CKPT_DEFAULT = "checkpoints/noise_sig_experiments/noise_sig_tres_abs_merged/best.ckpt"
TRES_MEAN, TRES_STD = 7.8, 26.2


def load_norm(h5_path):
    with h5py.File(h5_path, "r") as f:
        return (
            f["norm_param/mean"][:].astype(np.float32),
            f["norm_param/std"][:].astype(np.float32),
        )


def load_model(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model = EncoderTwoHead(
        in_features=5,
        hidden_size=128,
        num_shared_layers=5,
        num_cls_layers=0,
        num_tres_layers=0,
        dim_feedforward_size=512,
        n_heads=1,
        cls_out_size=2,
        tres_out_size=1,
        dropout_p=0.0,
    )
    # checkpoint t_res head is an MLP (128 -> 256 -> 1) with GELU
    model.tres_head = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 1))
    model.load_state_dict(sd, strict=True)
    return model.to(DEVICE).eval()


def infer_h5(model, h5_path, split, indices, renorm_fn, batch_size=128):
    """Infer events -> list of dicts: probs, channels, tres_pred, time_raw, q_raw."""
    events = []
    with h5py.File(h5_path, "r") as f:
        ev_starts = f[f"{split}/ev_starts/data"][:]
        data_all = f[f"{split}/data/data"]
        channels_all = f[f"{split}/channels/data"]
        norm_mean, norm_std = load_norm(h5_path)

        for bs in range(0, len(indices), batch_size):
            batch_idx = indices[bs : bs + batch_size]
            hits_list, ch_list, raw_list = [], [], []
            for idx in batch_idx:
                s, e = int(ev_starts[idx]), int(ev_starts[idx + 1])
                hits = data_all[s:e].astype(np.float32)
                raw = hits * norm_std + norm_mean
                if renorm_fn is not None:
                    hits = renorm_fn(hits)
                hits_list.append(hits)
                ch_list.append(channels_all[s:e])
                raw_list.append(raw)

            max_len = max(len(h) for h in hits_list)
            B = len(hits_list)
            x = np.zeros((B, max_len, 5), dtype=np.float32)
            mask = np.zeros((B, max_len), dtype=np.float32)
            for i, h in enumerate(hits_list):
                x[i, : len(h)] = h
                mask[i, : len(h)] = 1.0

            with torch.no_grad():
                out = model(torch.tensor(x).to(DEVICE), torch.tensor(mask).to(DEVICE).bool())
            probs = torch.sigmoid(out[:, :, 1]).cpu().numpy()
            tres_pred = out[:, :, 2].cpu().numpy() * TRES_STD + TRES_MEAN

            for i in range(B):
                L = len(hits_list[i])
                events.append(
                    {
                        "probs": probs[i][:L],
                        "channels": np.asarray(ch_list[i]),
                        "tres_pred": tres_pred[i][:L],
                        "time_raw": raw_list[i][:, FEAT_TIME],
                        "q_raw": raw_list[i][:, FEAT_Q],
                    }
                )
    return events


def event_passes_cut(ev, thr=0.5, min_hits=8, min_strings=2):
    sig = ev["probs"] > thr
    if int(sig.sum()) < min_hits:
        return False
    return len(np.unique(ev["channels"][sig] // 36)) >= min_strings


def collect_after_cut(model, h5_path, split, indices, renorm_fn, target, batch_size=128):
    kept = []
    chunk = 4096
    for i in tqdm(
        range(0, len(indices), chunk), desc=f"Collecting {Path(h5_path).name}", unit="chunk"
    ):
        idx = indices[i : i + chunk]
        for e in infer_h5(model, h5_path, split, idx, renorm_fn, batch_size):
            if event_passes_cut(e, 0.5):
                kept.append(e)
                if len(kept) >= target:
                    return kept
    return kept


def _signal_arrays(evs, thr, q_min, key):
    """Concatenate per-hit `key` over hits with prob>thr AND q>q_min."""
    out = []
    for e in evs:
        sel = (e["probs"] > thr) & (e["q_raw"] > q_min)
        out.append(e[key][sel])
    return np.concatenate(out) if out else np.array([])


def plot_row(axes, mc_evs, exp_evs, thr, q_min):
    FS_TITLE, FS_LABEL, FS_TICK = 12, 11, 10
    kw = dict(histtype="step", linewidth=1.8)

    mc_time = _signal_arrays(mc_evs, thr, q_min, "time_raw")
    exp_time = _signal_arrays(exp_evs, thr, q_min, "time_raw")
    mc_tres = _signal_arrays(mc_evs, thr, q_min, "tres_pred")
    exp_tres = _signal_arrays(exp_evs, thr, q_min, "tres_pred")

    # (col 0) raw hit time
    ax = axes[0]
    all_t = np.concatenate([mc_time, exp_time])
    lo, hi = np.percentile(all_t, [0.5, 99.5])
    bins_t = np.linspace(lo, hi, 61)
    ax.hist(mc_time, bins=bins_t, color=MC_COL, label=f"MC EAS ({mc_time.size})", **kw)
    ax.hist(exp_time, bins=bins_t, color=EXP_COL, label=f"EXP ({exp_time.size})", **kw)
    ax.set_yscale("log")
    ax.set_xlabel("Hit time [ns]", fontsize=FS_LABEL)
    ax.set_ylabel("Signal hits", fontsize=FS_LABEL)
    ax.set_title(rf"Hit time ($\xi>{thr:g}$, $Q>{q_min:g}$)", fontsize=FS_TITLE)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.3)

    # (col 1) predicted |t_res|
    ax = axes[1]
    bins_r = np.linspace(0, 100, 61)
    ax.hist(mc_tres, bins=bins_r, color=MC_COL, label=f"MC EAS ({mc_tres.size})", **kw)
    ax.hist(exp_tres, bins=bins_r, color=EXP_COL, label=f"EXP ({exp_tres.size})", **kw)
    ax.set_yscale("log")
    ax.set_xlabel(r"predicted $|t_{\mathrm{res}}|$ [ns]", fontsize=FS_LABEL)
    ax.set_ylabel("Signal hits", fontsize=FS_LABEL)
    ax.set_title(
        rf"$|t_{{\mathrm{{res}}}}|$ pred ($\xi>{thr:g}$, $Q>{q_min:g}$)", fontsize=FS_TITLE
    )
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.3)


def plot_qdist(axes, mc_evs, exp_evs, thresholds, q_hi=None):
    """One panel per threshold: raw Q distribution of signal hits (prob>thr)."""
    FS_TITLE, FS_LABEL, FS_TICK = 12, 11, 10
    kw = dict(histtype="step", linewidth=1.8)

    if q_hi is None:
        allq = _signal_arrays(mc_evs + exp_evs, min(thresholds), -np.inf, "q_raw")
        q_hi = float(np.percentile(allq, 99.5))
    bins = np.linspace(0, q_hi, 61)

    for ax, thr in zip(axes, thresholds):
        mc_q = _signal_arrays(mc_evs, thr, -np.inf, "q_raw")
        exp_q = _signal_arrays(exp_evs, thr, -np.inf, "q_raw")
        ax.hist(
            mc_q, bins=bins, color=MC_COL, label=f"MC EAS ({mc_q.size}, μ={mc_q.mean():.1f})", **kw
        )
        ax.hist(
            exp_q, bins=bins, color=EXP_COL, label=f"EXP ({exp_q.size}, μ={exp_q.mean():.1f})", **kw
        )
        ax.set_yscale("log")
        ax.set_xlabel("Q [p.e.]", fontsize=FS_LABEL)
        ax.set_ylabel("Signal hits", fontsize=FS_LABEL)
        ax.set_title(rf"Charge of signal hits ($\xi>{thr:g}$)", fontsize=FS_TITLE)
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--mc", default=MC_DEFAULT)
    ap.add_argument("--exp", default=EXP_DEFAULT)
    ap.add_argument("--target", type=int, default=5000, help="equal number of events after 8-2 cut")
    ap.add_argument(
        "--what",
        choices=["time_tres", "q"],
        default="time_tres",
        help="time_tres: hit time + |t_res| (rows=thresholds); "
        "q: charge distribution (one panel per threshold)",
    )
    ap.add_argument(
        "--q-min", type=float, default=1.0, help="charge cut for the time_tres histograms"
    )
    ap.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help="classification thresholds; default 0.5,0.8 for time_tres and 0.5,0.75,0.9 for q",
    )
    ap.add_argument(
        "--q-hi",
        type=float,
        default=None,
        help="upper Q axis limit for --what q (default 99.5 pct)",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    if args.thresholds is None:
        args.thresholds = [0.5, 0.8] if args.what == "time_tres" else [0.5, 0.75, 0.9]
    if args.out is None:
        args.out = (
            "plots/baseline_vs_hard/time_tres_q1.png"
            if args.what == "time_tres"
            else "plots/baseline_vs_hard/q_distribution.png"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = load_model(args.ckpt)

    mc_mean, mc_std = load_norm(args.mc)
    exp_mean, exp_std = load_norm(args.exp)

    def renorm_exp(h):
        return (h * exp_std + exp_mean - mc_mean) / mc_std

    with h5py.File(args.mc, "r") as f:
        ev_ids = f["train/ev_ids/data"][:]
    mc_idx = np.array(
        [i for i, eid in enumerate(ev_ids) if eid.decode().startswith("muatm")], dtype=np.int64
    )
    np.random.default_rng(42).shuffle(mc_idx)

    with h5py.File(args.exp, "r") as f:
        n_exp = len(f["train/ev_starts/data"]) - 1
    exp_idx = np.arange(n_exp, dtype=np.int64)
    np.random.default_rng(42).shuffle(exp_idx)

    print(f"Collecting {args.target} MC EAS events after 8-2 cut...")
    mc_evs = collect_after_cut(model, args.mc, "train", mc_idx, None, args.target, args.batch_size)
    print(f"  got {len(mc_evs)}")
    print(f"Collecting {args.target} EXP events after 8-2 cut...")
    exp_evs = collect_after_cut(
        model, args.exp, "train", exp_idx, renorm_exp, args.target, args.batch_size
    )
    print(f"  got {len(exp_evs)}")

    n = min(len(mc_evs), len(exp_evs))
    mc_evs, exp_evs = mc_evs[:n], exp_evs[:n]
    print(f"Using {n} events each")

    n_thr = len(args.thresholds)
    if args.what == "q":
        fig, axes = plt.subplots(1, n_thr, figsize=(5.5 * n_thr, 4.6), constrained_layout=True)
        axes = np.atleast_1d(axes)
        plot_qdist(axes, mc_evs, exp_evs, args.thresholds, args.q_hi)
        fig.suptitle(
            f"MC EAS vs EXP — signal-hit charge, raw counts, {n} events each, "
            f"8-2 cut [merged tres model]",
            fontsize=14,
        )
    else:
        fig, axes = plt.subplots(n_thr, 2, figsize=(11, 4.6 * n_thr), constrained_layout=True)
        if n_thr == 1:
            axes = axes[np.newaxis, :]
        for row, thr in enumerate(args.thresholds):
            plot_row(axes[row], mc_evs, exp_evs, thr, args.q_min)
        fig.suptitle(
            f"MC EAS vs EXP — raw counts, {n} events each, 8-2 cut, "
            f"Q>{args.q_min:g} [merged tres model]",
            fontsize=14,
        )
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
