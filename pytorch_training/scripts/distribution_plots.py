"""Comprehensive hit- and event-level distribution plots: MC val vs EXP.

Features: [charge, time, x, y, z] (indices 0-4).
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.encoder import Encoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MC_2020 = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_2020_sig-noise_mid-eq_normed.h5"
EXP_DATA = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_exp_filtered_norm_w_model_enc_v2_fix_q.h5"
CKPT = "checkpoints/noise_sig_rerun_v2/encoder_nl5_nh1_dff128_hs128_bs128/best_2020.ckpt"

FEAT_CHARGE = 0
FEAT_Z = 4

OUTPUT_DIR = Path("plots/distributions")


def load_norm_params(h5_path):
    with h5py.File(h5_path, "r") as f:
        mean = f["norm_param/mean"][:].astype(np.float32)
        std = f["norm_param/std"][:].astype(np.float32)
    return mean, std


def renormalize(data, src_mean, src_std, dst_mean, dst_std):
    return (data * src_std + src_mean - dst_mean) / dst_std


def denormalize(data, mean, std):
    return data * std + mean


def load_events(h5_path, split, max_events, batch_size=128, filter_muatm=False, renorm_fn=None):
    """Yield (x_normed, x_raw, mask, channels) batches.

    x_normed: for model input (possibly renormalized).
    x_raw: denormalized with source norms (for physical distributions).
    """
    src_mean, src_std = load_norm_params(h5_path)

    with h5py.File(h5_path, "r") as f:
        ev_starts = f[f"{split}/ev_starts/data"][:]
        channels_all = f[f"{split}/channels/data"]
        data_all = f[f"{split}/data/data"]
        n_events = len(ev_starts) - 1

        if filter_muatm:
            ev_ids = f[f"{split}/ev_ids/data"][:]
            indices = np.array([i for i in range(n_events) if ev_ids[i].startswith(b"muatm")])
            if max_events:
                indices = indices[:max_events]
        else:
            indices = np.arange(min(max_events, n_events) if max_events else n_events)

        for batch_start in range(0, len(indices), batch_size):
            batch_idx = indices[batch_start : batch_start + batch_size]
            events_norm, events_raw, ch_events = [], [], []
            for idx in batch_idx:
                s, e = int(ev_starts[idx]), int(ev_starts[idx + 1])
                hits = data_all[s:e].astype(np.float32)
                raw = denormalize(hits, src_mean, src_std)
                if renorm_fn is not None:
                    hits_for_model = renorm_fn(hits)
                else:
                    hits_for_model = hits
                events_norm.append(hits_for_model)
                events_raw.append(raw)
                ch_events.append(channels_all[s:e])

            max_len = max(len(ev) for ev in events_norm)
            bs = len(events_norm)
            x_norm = np.zeros((bs, max_len, 5), dtype=np.float32)
            x_raw = np.zeros((bs, max_len, 5), dtype=np.float32)
            mask = np.zeros((bs, max_len), dtype=np.float32)
            ch_pad = np.zeros((bs, max_len), dtype=np.int32)
            for i, (en, er, ch) in enumerate(zip(events_norm, events_raw, ch_events)):
                x_norm[i, : len(en)] = en
                x_raw[i, : len(er)] = er
                mask[i, : len(en)] = 1.0
                ch_pad[i, : len(en)] = ch

            yield (torch.tensor(x_norm), torch.tensor(x_raw), torch.tensor(mask), ch_pad)


def load_model(ckpt_path, hidden_size=128, dff=128):
    model = Encoder(
        in_features=5,
        hidden_size=hidden_size,
        num_layers=5,
        dim_feedforward_size=dff,
        n_heads=1,
        out_size=2,
        dropout_p=0.0,
    )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    return model.to(DEVICE).eval()


def run_inference(model, data_iter, threshold=0.5):
    """Returns per-event dicts with hit-level info."""
    events = []
    model.eval()
    with torch.no_grad():
        for x_norm, x_raw, mask, channels in tqdm(data_iter, desc="Inference"):
            out = model(x_norm.to(DEVICE), mask.to(DEVICE).bool())
            probs = torch.sigmoid(out[:, :, 1]).cpu().numpy()
            mask_np = mask.numpy().astype(bool)
            raw_np = x_raw.numpy()
            for i in range(x_norm.shape[0]):
                m = mask_np[i]
                p = probs[i][m]
                ch = channels[i][m]
                charge = raw_np[i, m, FEAT_CHARGE]
                z = raw_np[i, m, FEAT_Z]
                events.append(
                    {
                        "probs": p,
                        "charge": charge,
                        "z": z,
                        "channels": ch,
                    }
                )
    return events


def apply_threshold_and_cuts(events, threshold, min_hits=0, min_strings=0):
    """For each event, determine signal hits and apply event-level cuts.

    Returns:
        passed_events: list of dicts with signal hit info for events passing cuts
        n_pass, n_total: counts
    """
    passed = []
    for ev in events:
        sig_mask = ev["probs"] > threshold
        n_sig = sig_mask.sum()
        if n_sig > 0:
            sig_strings = np.unique(ev["channels"][sig_mask] // 36)
            n_str = len(sig_strings)
        else:
            n_str = 0

        if n_sig >= min_hits and n_str >= min_strings:
            passed.append(
                {
                    "all_probs": ev["probs"],
                    "sig_charge": ev["charge"][sig_mask],
                    "sig_z": ev["z"][sig_mask],
                    "n_sig_hits": int(n_sig),
                    "sig_charge_sum": float(ev["charge"][sig_mask].sum()) if n_sig > 0 else 0.0,
                }
            )
    return passed, len(passed), len(events)


def make_plots(mc_events, exp_events, threshold, cuts_list, output_dir):
    """Generate distribution plots for each cut configuration."""

    for min_hits, min_strings in cuts_list:
        cut_label = (
            f"≥{min_hits}h ≥{min_strings}s" if (min_hits > 0 or min_strings > 0) else "no cuts"
        )
        cut_suffix = (
            f"{min_hits}h_{min_strings}s" if (min_hits > 0 or min_strings > 0) else "nocuts"
        )

        mc_pass, mc_np, mc_nt = apply_threshold_and_cuts(
            mc_events, threshold, min_hits, min_strings
        )
        exp_pass, exp_np, exp_nt = apply_threshold_and_cuts(
            exp_events, threshold, min_hits, min_strings
        )

        print(f"\n  Cut {cut_label}: MC {mc_np}/{mc_nt} ev, EXP {exp_np}/{exp_nt} ev")

        if mc_np == 0 or exp_np == 0:
            print("  Skipping — no events pass cuts")
            continue

        mc_all_probs = np.concatenate([e["all_probs"] for e in mc_pass])
        exp_all_probs = np.concatenate([e["all_probs"] for e in exp_pass])
        mc_sig_charge = np.concatenate([e["sig_charge"] for e in mc_pass])
        exp_sig_charge = np.concatenate([e["sig_charge"] for e in exp_pass])
        mc_sig_z = np.concatenate([e["sig_z"] for e in mc_pass])
        exp_sig_z = np.concatenate([e["sig_z"] for e in exp_pass])
        mc_nhits = np.array([e["n_sig_hits"] for e in mc_pass])
        exp_nhits = np.array([e["n_sig_hits"] for e in exp_pass])
        mc_qsum = np.array([e["sig_charge_sum"] for e in mc_pass])
        exp_qsum = np.array([e["sig_charge_sum"] for e in exp_pass])

        fig, axes = plt.subplots(3, 2, figsize=(16, 16))
        mc_lbl = f"MC muon ({mc_np}/{mc_nt} ev)"
        exp_lbl = f"EXP ({exp_np}/{exp_nt} ev)"
        mc_col, exp_col = "#2ca02c", "#1f77b4"

        # (0,0) Score distribution, log y
        ax = axes[0, 0]
        bins_score = np.linspace(0, 1, 80)
        ax.hist(
            mc_all_probs,
            bins=bins_score,
            density=True,
            histtype="step",
            linewidth=2,
            color=mc_col,
            label=mc_lbl,
        )
        ax.hist(
            exp_all_probs,
            bins=bins_score,
            density=True,
            histtype="step",
            linewidth=2,
            color=exp_col,
            label=exp_lbl,
        )
        ax.set_yscale("log")
        ax.set_xlabel("P(signal)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        wd_score = wasserstein_distance(mc_all_probs, exp_all_probs)
        ax.set_title(f"Score distribution (W={wd_score:.4f})", fontsize=12)
        ax.axvline(
            threshold, color="red", ls="--", lw=1.5, alpha=0.7, label=f"threshold={threshold}"
        )
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle=":")

        # (0,1) Charge distribution on signal hits
        ax = axes[0, 1]
        q_max = np.percentile(np.concatenate([mc_sig_charge, exp_sig_charge]), 99)
        bins_q = np.linspace(0, q_max, 60)
        ax.hist(
            mc_sig_charge,
            bins=bins_q,
            density=True,
            histtype="step",
            linewidth=2,
            color=mc_col,
            label=mc_lbl,
        )
        ax.hist(
            exp_sig_charge,
            bins=bins_q,
            density=True,
            histtype="step",
            linewidth=2,
            color=exp_col,
            label=exp_lbl,
        )
        ax.set_xlabel("Charge (signal hits)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        wd_q = wasserstein_distance(mc_sig_charge, exp_sig_charge)
        ax.set_title(f"Charge on signal hits (W={wd_q:.2f})", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle=":")

        # (1,0) Z distribution (unweighted)
        ax = axes[1, 0]
        bins_z = np.linspace(-300, 300, 60)
        ax.hist(
            mc_sig_z,
            bins=bins_z,
            density=True,
            histtype="step",
            linewidth=2,
            color=mc_col,
            label=mc_lbl,
        )
        ax.hist(
            exp_sig_z,
            bins=bins_z,
            density=True,
            histtype="step",
            linewidth=2,
            color=exp_col,
            label=exp_lbl,
        )
        ax.set_xlabel("Z (signal hits)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        wd_z = wasserstein_distance(mc_sig_z, exp_sig_z)
        ax.set_title(f"Z distribution — unweighted (W={wd_z:.1f})", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle=":")

        # (1,1) Z distribution (charge-weighted)
        ax = axes[1, 1]
        ax.hist(
            mc_sig_z,
            bins=bins_z,
            weights=mc_sig_charge,
            density=True,
            histtype="step",
            linewidth=2,
            color=mc_col,
            label=mc_lbl,
        )
        ax.hist(
            exp_sig_z,
            bins=bins_z,
            weights=exp_sig_charge,
            density=True,
            histtype="step",
            linewidth=2,
            color=exp_col,
            label=exp_lbl,
        )
        ax.set_xlabel("Z (signal hits)", fontsize=12)
        ax.set_ylabel("Density (charge-weighted)", fontsize=12)
        ax.set_title("Z distribution — charge-weighted", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle=":")

        # (2,0) Signal hit multiplicity
        ax = axes[2, 0]
        nh_max = int(np.percentile(np.concatenate([mc_nhits, exp_nhits]), 99))
        bins_nh = np.arange(0, nh_max + 2) - 0.5
        ax.hist(
            mc_nhits,
            bins=bins_nh,
            density=True,
            histtype="step",
            linewidth=2,
            color=mc_col,
            label=mc_lbl,
        )
        ax.hist(
            exp_nhits,
            bins=bins_nh,
            density=True,
            histtype="step",
            linewidth=2,
            color=exp_col,
            label=exp_lbl,
        )
        ax.set_xlabel("N signal hits per event", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        wd_nh = wasserstein_distance(mc_nhits, exp_nhits)
        ax.set_title(f"Signal hit multiplicity (W={wd_nh:.2f})", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle=":")

        # (2,1) Charge sum per event
        ax = axes[2, 1]
        qs_max = np.percentile(np.concatenate([mc_qsum, exp_qsum]), 99)
        bins_qs = np.linspace(0, qs_max, 60)
        ax.hist(
            mc_qsum,
            bins=bins_qs,
            density=True,
            histtype="step",
            linewidth=2,
            color=mc_col,
            label=mc_lbl,
        )
        ax.hist(
            exp_qsum,
            bins=bins_qs,
            density=True,
            histtype="step",
            linewidth=2,
            color=exp_col,
            label=exp_lbl,
        )
        ax.set_xlabel("Total charge of signal hits per event", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        wd_qs = wasserstein_distance(mc_qsum, exp_qsum)
        ax.set_title(f"Charge sum per event (W={wd_qs:.2f})", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle=":")

        fig.suptitle(
            f"MC muon vs EXP — threshold={threshold}, {cut_label}",
            fontsize=15,
            fontweight="bold",
        )
        plt.tight_layout()
        out_path = output_dir / f"distributions_t{threshold}_{cut_suffix}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-events", type=int, default=10_000)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--mc-split", default="val")
    parser.add_argument("--exp-split", default="train")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exp_mean, exp_std = load_norm_params(EXP_DATA)
    mc_mean, mc_std = load_norm_params(MC_2020)

    def renorm_exp_to_mc(data):
        return renormalize(data, exp_mean, exp_std, mc_mean, mc_std)

    print("Loading MC muon events...")
    mc_batches = list(load_events(MC_2020, args.mc_split, args.max_events, filter_muatm=True))
    n_mc = sum(x.shape[0] for x, _, _, _ in mc_batches)
    print(f"  {n_mc} MC muon events")

    print("Loading EXP events...")
    exp_batches = list(
        load_events(EXP_DATA, args.exp_split, args.max_events, renorm_fn=renorm_exp_to_mc)
    )
    n_exp = sum(x.shape[0] for x, _, _, _ in exp_batches)
    print(f"  {n_exp} EXP events")

    print("Loading model...")
    model = load_model(CKPT, hidden_size=128, dff=128)

    print("Running inference on MC...")
    mc_events = run_inference(model, iter(mc_batches))
    print(f"  {len(mc_events)} events processed")

    print("Running inference on EXP...")
    exp_events = run_inference(model, iter(exp_batches))
    print(f"  {len(exp_events)} events processed")

    del model
    torch.cuda.empty_cache()

    cuts_list = [(0, 0), (8, 2), (8, 3), (10, 3)]

    for thr in args.thresholds:
        print(f"\n{'=' * 60}")
        print(f"Threshold = {thr}")
        print(f"{'=' * 60}")
        make_plots(mc_events, exp_events, thr, cuts_list, OUTPUT_DIR)


if __name__ == "__main__":
    main()
