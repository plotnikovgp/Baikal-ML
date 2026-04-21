"""Plot per-event mean Z distributions: all hits vs signal hits, MC vs EXP."""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.encoder import Encoder, EncoderDomainAdaptation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MC_2020 = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_2020_sig-noise_mid-eq_normed.h5"
EXP_DATA = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_exp_filtered_norm_w_model_enc_v2_fix_q.h5"

FEAT_Z = 4
MC_COL = "#2ca02c"
EXP_COL = "#1f77b4"

plt.rcParams.update({"font.size": 12, "figure.dpi": 130})


def load_norm_params(h5_path):
    with h5py.File(h5_path, "r") as f:
        return f["norm_param/mean"][:].astype(np.float32), f["norm_param/std"][:].astype(np.float32)


def denormalize(data, mean, std):
    return data * std + mean


def renormalize(data, src_mean, src_std, dst_mean, dst_std):
    return (data * src_std + src_mean - dst_mean) / dst_std


def load_model(ckpt_path, model_type="encoder", hs=128, dff=128):
    if model_type == "encoder_da":
        model = EncoderDomainAdaptation(
            in_features=5,
            hidden_size=hs,
            num_layers=5,
            dim_feedforward_size=dff,
            n_heads=1,
            out_size=2,
            dropout_p=0.0,
            num_domains=2,
            domain_classifier_hidden_size=64,
            domain_classifier_layers=2,
            gradient_reversal_alpha=1.0,
            aggregate_output=False,
        )
    else:
        model = Encoder(
            in_features=5,
            hidden_size=hs,
            num_layers=5,
            dim_feedforward_size=dff,
            n_heads=1,
            out_size=2,
            dropout_p=0.0,
        )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    return model.to(DEVICE).eval()


def model_forward(model, x, mask):
    out = model(x, mask)
    if isinstance(out, tuple):
        out = out[0]
    return out


def process_mc(h5_path, split, max_events, model, ev_type_prefix="muatm", threshold=0.5):
    """Return per-event mean Z for all hits and signal hits (by label and by model)."""
    src_mean, src_std = load_norm_params(h5_path)
    mean_z_all, mean_z_sig_label = [], []
    mean_z_sig_model = []

    with h5py.File(h5_path, "r") as f:
        ev_starts = f[f"{split}/ev_starts/data"][:]
        data_all = f[f"{split}/data/data"]
        labels_all = f[f"{split}/labels/data"]
        ev_ids = f[f"{split}/ev_ids/data"][:]
        n_total = len(ev_starts) - 1

        type_idx = [i for i in range(n_total) if ev_ids[i].decode().startswith(ev_type_prefix)]
        sel = type_idx[:max_events]
        print(f"  MC {ev_type_prefix}: {len(sel)} events")

        batch_size = 128
        for bs_start in tqdm(range(0, len(sel), batch_size), desc=f"MC {ev_type_prefix}"):
            batch_idx = sel[bs_start : bs_start + batch_size]
            evs_raw_z, evs_labels, evs_norm = [], [], []
            for idx in batch_idx:
                s, e = int(ev_starts[idx]), int(ev_starts[idx + 1])
                hits = data_all[s:e].astype(np.float32)
                raw = denormalize(hits, src_mean, src_std)
                evs_raw_z.append(raw[:, FEAT_Z])
                evs_labels.append(labels_all[s:e])
                evs_norm.append(hits)

            max_len = max(len(ev) for ev in evs_norm)
            bs = len(evs_norm)
            x_n = np.zeros((bs, max_len, 5), dtype=np.float32)
            mask = np.zeros((bs, max_len), dtype=np.float32)
            for i in range(bs):
                L = len(evs_norm[i])
                x_n[i, :L] = evs_norm[i]
                mask[i, :L] = 1.0

            with torch.no_grad():
                out = model_forward(
                    model, torch.tensor(x_n).to(DEVICE), torch.tensor(mask).to(DEVICE).bool()
                )
                probs = torch.sigmoid(out[:, :, 1]).cpu().numpy()

            for i in range(bs):
                z = evs_raw_z[i]
                lab = evs_labels[i]
                L = len(z)
                p = probs[i, :L]

                mean_z_all.append(np.mean(z))
                sig_mask_label = lab != 0
                if sig_mask_label.sum() > 0:
                    mean_z_sig_label.append(np.mean(z[sig_mask_label]))
                sig_mask_model = p > threshold
                if sig_mask_model.sum() > 0:
                    mean_z_sig_model.append(np.mean(z[sig_mask_model]))

    return np.array(mean_z_all), np.array(mean_z_sig_label), np.array(mean_z_sig_model)


def process_exp(h5_path, split, max_events, model, renorm_fn, threshold=0.5):
    """Return per-event mean Z for all hits and signal hits (by model)."""
    src_mean, src_std = load_norm_params(h5_path)
    mean_z_all, mean_z_sig = [], []

    with h5py.File(h5_path, "r") as f:
        ev_starts = f[f"{split}/ev_starts/data"][:]
        data_all = f[f"{split}/data/data"]
        n = min(max_events, len(ev_starts) - 1)
        print(f"  EXP: {n} events")

        batch_size = 128
        for bs_start in tqdm(range(0, n, batch_size), desc="EXP"):
            bs_end = min(bs_start + batch_size, n)
            evs_raw_z, evs_norm = [], []
            for idx in range(bs_start, bs_end):
                s, e = int(ev_starts[idx]), int(ev_starts[idx + 1])
                hits = data_all[s:e].astype(np.float32)
                raw = denormalize(hits, src_mean, src_std)
                evs_raw_z.append(raw[:, FEAT_Z])
                evs_norm.append(renorm_fn(hits))

            max_len = max(len(ev) for ev in evs_norm)
            bs = len(evs_norm)
            x_n = np.zeros((bs, max_len, 5), dtype=np.float32)
            mask = np.zeros((bs, max_len), dtype=np.float32)
            for i in range(bs):
                L = len(evs_norm[i])
                x_n[i, :L] = evs_norm[i]
                mask[i, :L] = 1.0

            with torch.no_grad():
                out = model_forward(
                    model, torch.tensor(x_n).to(DEVICE), torch.tensor(mask).to(DEVICE).bool()
                )
                probs = torch.sigmoid(out[:, :, 1]).cpu().numpy()

            for i in range(bs):
                z = evs_raw_z[i]
                L = len(z)
                p = probs[i, :L]
                mean_z_all.append(np.mean(z))
                sig_mask = p > threshold
                if sig_mask.sum() > 0:
                    mean_z_sig.append(np.mean(z[sig_mask]))

    return np.array(mean_z_all), np.array(mean_z_sig)


def plot_mean_z(
    mc_all,
    mc_sig_label,
    mc_sig_model,
    exp_all,
    exp_sig,
    ev_type_label,
    out_path,
    threshold,
    model_name="",
):
    bins = np.linspace(-300, 300, 40)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.hist(
        mc_all,
        bins=bins,
        density=True,
        histtype="step",
        lw=2,
        color=MC_COL,
        label=f"MC {ev_type_label} ({len(mc_all)} ev)",
    )
    ax.hist(
        exp_all,
        bins=bins,
        density=True,
        histtype="step",
        lw=2,
        color=EXP_COL,
        label=f"EXP ({len(exp_all)} ev)",
    )
    ax.set_xlabel("Mean Z per event [m]")
    ax.set_ylabel("Density")
    ax.set_title("All hits — per-event mean Z")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(
        mc_sig_label,
        bins=bins,
        density=True,
        histtype="step",
        lw=2,
        color=MC_COL,
        label=f"MC {ev_type_label} (label, {len(mc_sig_label)} ev)",
    )
    ax.hist(
        mc_sig_model,
        bins=bins,
        density=True,
        histtype="step",
        lw=2,
        color=MC_COL,
        label=f"MC {ev_type_label} (model, {len(mc_sig_model)} ev)",
        ls="--",
    )
    ax.hist(
        exp_sig,
        bins=bins,
        density=True,
        histtype="step",
        lw=2,
        color=EXP_COL,
        label=f"EXP (model, {len(exp_sig)} ev)",
    )
    ax.set_xlabel("Mean Z per event [m]")
    ax.set_ylabel("Density")
    ax.set_title(f"Signal hits (threshold={threshold}) — per-event mean Z")
    ax.legend()
    ax.grid(True, alpha=0.3)

    title = f"Per-event mean Z: MC {ev_type_label} vs EXP  (10k events)"
    if model_name:
        title += f"\nModel: {model_name}"
    fig.suptitle(title, fontweight="bold", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", default="encoder", choices=["encoder", "encoder_da"])
    parser.add_argument("--model-name", default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--hs", type=int, default=128)
    parser.add_argument("--dff", type=int, default=128)
    parser.add_argument("--out-prefix", default="mean_z")
    parser.add_argument("--max-events", type=int, default=10_000)
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint} (type={args.model_type})...")
    model = load_model(args.checkpoint, model_type=args.model_type, hs=args.hs, dff=args.dff)

    exp_mean, exp_std = load_norm_params(EXP_DATA)
    mc_mean, mc_std = load_norm_params(MC_2020)

    def renorm_fn(d):
        return renormalize(d, exp_mean, exp_std, mc_mean, mc_std)

    print("Processing EXP events...")
    exp_all, exp_sig = process_exp(
        EXP_DATA, "train", args.max_events, model, renorm_fn, threshold=args.threshold
    )

    out_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    for prefix, label in [("muatm", "muon"), ("nuatm", "nuatm"), ("nue2", "nue2")]:
        print(f"\nProcessing MC {prefix} events...")
        mc_all, mc_sig_label, mc_sig_model = process_mc(
            MC_2020, "val", args.max_events, model, ev_type_prefix=prefix, threshold=args.threshold
        )
        plot_mean_z(
            mc_all,
            mc_sig_label,
            mc_sig_model,
            exp_all,
            exp_sig,
            label,
            out_dir / f"{args.out_prefix}_{prefix}.png",
            args.threshold,
            args.model_name,
        )


if __name__ == "__main__":
    main()
