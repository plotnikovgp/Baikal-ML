"""Compare P(signal) distributions of z-normal vs z-mirrored networks
on MC and EXP data."""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import ks_2samp, wasserstein_distance
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.encoder import Encoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MC_MERGED = "/home/plotnikovgp/baikal/data/baikal_mc_merged_m50-na15-nue15_norm.h5"
MC_2020 = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_2020_sig-noise_mid-eq_normed.h5"
EXP_DATA = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_exp_filtered_norm_w_model_enc_v2_fix_q.h5"
CKPT_A = "checkpoints/noise_sig_rerun_v2/encoder_nl5_nh1_dff128_hs128_bs128/best_2020.ckpt"
CKPT_B = "checkpoints/noise_sig_rerun_v2/encoder_nl5_nh1_dff512_hs512_bs128/best_2020.ckpt"

OUTPUT_DIR = Path("plots/z_mirror_comparison")


def load_norm_params(h5_path):
    with h5py.File(h5_path, "r") as f:
        mean = f["norm_param/mean"][:].astype(np.float32)
        std = f["norm_param/std"][:].astype(np.float32)
    return mean, std


def renormalize(data, src_mean, src_std, dst_mean, dst_std):
    return (data * src_std + src_mean - dst_mean) / dst_std


def load_events(h5_path, split, max_events, batch_size=128, filter_muatm=False, renorm_fn=None):
    """Yield (x, mask, channels) batches."""
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
            events, ch_events = [], []
            for idx in batch_idx:
                s, e = int(ev_starts[idx]), int(ev_starts[idx + 1])
                hits = data_all[s:e].astype(np.float32)
                if renorm_fn is not None:
                    hits = renorm_fn(hits)
                events.append(hits)
                ch_events.append(channels_all[s:e])

            max_len = max(len(ev) for ev in events)
            bs = len(events)
            x_pad = np.zeros((bs, max_len, 5), dtype=np.float32)
            mask = np.zeros((bs, max_len), dtype=np.float32)
            ch_pad = np.zeros((bs, max_len), dtype=np.int32)
            for i, (ev, ch) in enumerate(zip(events, ch_events)):
                x_pad[i, : len(ev)] = ev
                mask[i, : len(ev)] = 1.0
                ch_pad[i, : len(ev)] = ch

            yield torch.tensor(x_pad), torch.tensor(mask), ch_pad


def run_inference(model, data_iter, threshold=0.5):
    ev_probs, ev_nhits, ev_nstr = [], [], []
    model.eval()
    with torch.no_grad():
        for x, mask, channels in tqdm(data_iter, desc="Inference"):
            out = model(x.to(DEVICE), mask.to(DEVICE).bool())
            probs = torch.sigmoid(out[:, :, 1]).cpu().numpy()
            mask_np = mask.numpy().astype(bool)
            for i in range(x.shape[0]):
                m = mask_np[i]
                p = probs[i][m]
                ch = channels[i][m]
                preds = p > threshold
                ev_probs.append(p)
                ev_nhits.append(int(preds.sum()))
                ev_nstr.append(len(np.unique((ch // 36)[preds])))
    return ev_probs, np.array(ev_nhits), np.array(ev_nstr)


def apply_cuts(ev_probs, nhits, nstr, min_hits=0, min_strings=0):
    mask = (nhits >= min_hits) & (nstr >= min_strings)
    if not mask.any():
        return np.array([]), 0, len(mask)
    flat = np.concatenate([ev_probs[i] for i in np.where(mask)[0]])
    return flat, int(mask.sum()), len(mask)


def load_model(ckpt_path, hidden_size=128, dim_feedforward_size=512):
    model = Encoder(
        in_features=5,
        hidden_size=hidden_size,
        num_layers=5,
        dim_feedforward_size=dim_feedforward_size,
        n_heads=1,
        out_size=2,
        dropout_p=0.0,
    )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    return model.to(DEVICE).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-events", type=int, default=10_000)
    parser.add_argument("--min-hits", type=int, default=0)
    parser.add_argument("--min-strings", type=int, default=0)
    parser.add_argument("--split", default="val")
    parser.add_argument("--exp-split", default="train")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exp_mean, exp_std = load_norm_params(EXP_DATA)
    mc_merged_mean, mc_merged_std = load_norm_params(MC_MERGED)
    mc_2020_mean, mc_2020_std = load_norm_params(MC_2020)

    def renorm_exp_to_mc_merged(data):
        return renormalize(data, exp_mean, exp_std, mc_merged_mean, mc_merged_std)

    def renorm_exp_to_mc_2020(data):
        return renormalize(data, exp_mean, exp_std, mc_2020_mean, mc_2020_std)

    # Model A: hs=128, trained on MC 2020
    print("\nLoading EXP for model A (renorm to MC 2020)...")
    exp_batches_a = list(
        load_events(EXP_DATA, args.exp_split, args.max_events, renorm_fn=renorm_exp_to_mc_2020)
    )
    print(f"  {sum(x.shape[0] for x, _, _ in exp_batches_a)} events")

    print("Running model A (hs=128) on EXP...")
    model_a = load_model(CKPT_A, hidden_size=128, dim_feedforward_size=128)
    ep_a, nh_a, ns_a = run_inference(model_a, iter(exp_batches_a))
    probs_a, pass_a, tot_a = apply_cuts(ep_a, nh_a, ns_a, args.min_hits, args.min_strings)
    print(f"  {pass_a}/{tot_a} ev pass, {len(probs_a):,} hits")
    del model_a, exp_batches_a
    torch.cuda.empty_cache()

    # Model B: hs=512, trained on MC 2020
    print("\nLoading EXP for model B (renorm to MC 2020)...")
    exp_batches_b = list(
        load_events(EXP_DATA, args.exp_split, args.max_events, renorm_fn=renorm_exp_to_mc_2020)
    )
    print(f"  {sum(x.shape[0] for x, _, _ in exp_batches_b)} events")

    print("Running model B (hs=512) on EXP...")
    model_b = load_model(CKPT_B, hidden_size=512)
    ep_b, nh_b, ns_b = run_inference(model_b, iter(exp_batches_b))
    probs_d, pass_b, tot_b = apply_cuts(ep_b, nh_b, ns_b, args.min_hits, args.min_strings)
    print(f"  {pass_b}/{tot_b} ev pass, {len(probs_d):,} hits")
    del model_b, exp_batches_b
    torch.cuda.empty_cache()

    _exp_pass, _exp_tot = pass_a, tot_a

    # MC muon events (no renorm needed — both models trained on MC_2020)
    print(f"\nLoading MC 2020 muon events ({args.split} split)...")
    mc_batches = list(load_events(MC_2020, args.split, args.max_events, filter_muatm=True))
    n_mc = sum(x.shape[0] for x, _, _ in mc_batches)
    print(f"  {n_mc} muon events")

    print("Running model A (hs=128) on MC muon...")
    model_a = load_model(CKPT_A, hidden_size=128, dim_feedforward_size=128)
    mc_ep_a, mc_nh_a, mc_ns_a = run_inference(model_a, iter(mc_batches))
    mc_probs_a, mc_pass_a, mc_tot_a = apply_cuts(
        mc_ep_a, mc_nh_a, mc_ns_a, args.min_hits, args.min_strings
    )
    print(f"  {mc_pass_a}/{mc_tot_a} ev pass, {len(mc_probs_a):,} hits")
    del model_a
    torch.cuda.empty_cache()

    print("Running model B (hs=512) on MC muon...")
    model_b = load_model(CKPT_B, hidden_size=512)
    mc_ep_b, mc_nh_b, mc_ns_b = run_inference(model_b, iter(mc_batches))
    mc_probs_b, mc_pass_b, mc_tot_b = apply_cuts(
        mc_ep_b, mc_nh_b, mc_ns_b, args.min_hits, args.min_strings
    )
    print(f"  {mc_pass_b}/{mc_tot_b} ev pass, {len(mc_probs_b):,} hits")
    del model_b, mc_batches
    torch.cuda.empty_cache()

    def make_plot(
        exp_a, exp_b, mc_a, mc_b, exp_info_a, exp_info_b, mc_info_a, mc_info_b, cut_label, out_path
    ):
        eps = 1e-10
        le_a = np.log10(np.clip(exp_a, eps, None))
        le_b = np.log10(np.clip(exp_b, eps, None))
        lm_a = np.log10(np.clip(mc_a, eps, None))
        lm_b = np.log10(np.clip(mc_b, eps, None))
        wd_a = wasserstein_distance(exp_a, mc_a)
        wd_b = wasserstein_distance(exp_b, mc_b)

        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        bins = np.linspace(-4, 0, 80)

        ax = axes[0, 0]
        ax.hist(
            lm_a,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color="#2ca02c",
            label=f"MC muon ({len(mc_a):,} hits, {mc_info_a})",
        )
        ax.hist(
            le_a,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color="#1f77b4",
            label=f"EXP ({len(exp_a):,} hits, {exp_info_a})",
        )
        ax.set_xlabel("log₁₀ P(signal)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"No z-mirror — histogram (W={wd_a:.4f})", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle=":")
        ax.set_xlim(-4, 0)

        ax = axes[0, 1]
        ax.plot(
            np.sort(lm_a),
            np.linspace(0, 1, len(lm_a)),
            label="MC muon",
            linewidth=2,
            color="#2ca02c",
        )
        ax.plot(
            np.sort(le_a), np.linspace(0, 1, len(le_a)), label="EXP", linewidth=2, color="#1f77b4"
        )
        ax.set_xlabel("log₁₀ P(signal)", fontsize=12)
        ax.set_ylabel("CDF", fontsize=12)
        ax.set_title("No z-mirror — CDF", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, linestyle=":")
        ax.set_xlim(-4, 0)

        ax = axes[1, 0]
        ax.hist(
            lm_b,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color="#2ca02c",
            label=f"MC muon ({len(mc_b):,} hits, {mc_info_b})",
        )
        ax.hist(
            le_b,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color="#1f77b4",
            label=f"EXP ({len(exp_b):,} hits, {exp_info_b})",
        )
        ax.set_xlabel("log₁₀ P(signal)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Z-mirror — histogram (W={wd_b:.4f})", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle=":")
        ax.set_xlim(-4, 0)

        ax = axes[1, 1]
        ax.plot(
            np.sort(lm_b),
            np.linspace(0, 1, len(lm_b)),
            label="MC muon",
            linewidth=2,
            color="#2ca02c",
        )
        ax.plot(
            np.sort(le_b), np.linspace(0, 1, len(le_b)), label="EXP", linewidth=2, color="#1f77b4"
        )
        ax.set_xlabel("log₁₀ P(signal)", fontsize=12)
        ax.set_ylabel("CDF", fontsize=12)
        ax.set_title("Z-mirror — CDF", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, linestyle=":")
        ax.set_xlim(-4, 0)

        fig.suptitle(
            f"log₁₀ P(signal): MC muon vs EXP ({cut_label})", fontsize=15, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")
        print(f"  Wasserstein: no z-mirror = {wd_a:.6f}, z-mirror = {wd_b:.6f}")

    # No cuts
    make_plot(
        probs_a,
        probs_d,
        mc_probs_a,
        mc_probs_b,
        f"{pass_a}/{tot_a} ev",
        f"{pass_b}/{tot_b} ev",
        f"{mc_pass_a}/{mc_tot_a} ev",
        f"{mc_pass_b}/{mc_tot_b} ev",
        "no cuts",
        OUTPUT_DIR / "z_mirror_distributions.png",
    )

    # With 2-string 8-hit cuts
    min_h, min_s = 8, 2
    cut_exp_a, cut_ep_a, cut_et_a = apply_cuts(ep_a, nh_a, ns_a, min_h, min_s)
    cut_exp_b, cut_ep_b, cut_et_b = apply_cuts(ep_b, nh_b, ns_b, min_h, min_s)
    cut_mc_a, cut_mp_a, cut_mt_a = apply_cuts(mc_ep_a, mc_nh_a, mc_ns_a, min_h, min_s)
    cut_mc_b, cut_mp_b, cut_mt_b = apply_cuts(mc_ep_b, mc_nh_b, mc_ns_b, min_h, min_s)
    print(f"\nWith cuts (≥{min_h} hits, ≥{min_s} strings):")
    print(f"  EXP A: {cut_ep_a}/{cut_et_a} ev, {len(cut_exp_a):,} hits")
    print(f"  EXP B: {cut_ep_b}/{cut_et_b} ev, {len(cut_exp_b):,} hits")
    print(f"  MC  A: {cut_mp_a}/{cut_mt_a} ev, {len(cut_mc_a):,} hits")
    print(f"  MC  B: {cut_mp_b}/{cut_mt_b} ev, {len(cut_mc_b):,} hits")

    make_plot(
        cut_exp_a,
        cut_exp_b,
        cut_mc_a,
        cut_mc_b,
        f"{cut_ep_a}/{cut_et_a} ev",
        f"{cut_ep_b}/{cut_et_b} ev",
        f"{cut_mp_a}/{cut_mt_a} ev",
        f"{cut_mp_b}/{cut_mt_b} ev",
        f"≥{min_h} hits, ≥{min_s} strings",
        OUTPUT_DIR / "z_mirror_distributions_cut.png",
    )


if __name__ == "__main__":
    main()
