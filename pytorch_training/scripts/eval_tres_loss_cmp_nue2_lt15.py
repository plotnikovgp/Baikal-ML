"""Evaluate tres_loss_comparison checkpoints on nue2 validation, |true abs(tres)| < 15 ns.

Matches training: GT signal hits (label != 0), signal-only input, min_signal_hits=8.
Uses best_nue2.ckpt per experiment. Saves plots and metrics table under plots/tres_loss_cmp_eval_nue2_lt15/.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import r2_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.encoder import Encoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parent.parent
NUE2_H5 = "/home/plotnikovgp/baikal/data/merged_nue2.h5"
OUT_DIR = ROOT / "plots" / "tres_loss_cmp_eval_nue2_lt15"
CKPT_BASE = ROOT / "checkpoints" / "tres_loss_comparison"

TRES_MEAN = 7.8
TRES_STD = 26.2
MIN_SIGNAL_HITS = 8
TRUE_MAX_NS = 15.0

MODELS = [
    ("mae", "tres_merged_mae_hs128", "MAE"),
    ("mse", "tres_merged_mse_hs128", "MSE"),
    ("huber", "tres_merged_huber_hs128", "Huber"),
    ("mae_log", "tres_merged_mae_log_hs128", "MAE+log"),
]

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
    }
)


def load_encoder(ckpt: Path) -> Encoder:
    model = Encoder(
        in_features=5,
        hidden_size=128,
        num_layers=5,
        dim_feedforward_size=512,
        n_heads=1,
        out_size=1,
        dropout_p=0.0,
        head_bias=True,
    )
    sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    return model.to(DEVICE).eval()


def collect_predictions(
    model: Encoder,
    split: str = "val",
    max_events: int | None = None,
    batch_events: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pred_abs_ns, true_abs_ns for all GT signal hits in kept events."""
    pred_all = []
    true_all = []

    with h5py.File(NUE2_H5, "r") as f:
        ev_starts = f[f"{split}/ev_starts/data"][:]
        n_events = len(ev_starts) - 1
        if max_events is not None:
            n_events = min(n_events, max_events)

        batch_starts = []
        batch_ends = []
        cur = 0
        while cur < n_events:
            end = min(cur + batch_events, n_events)
            batch_starts.append(cur)
            batch_ends.append(end)
            cur = end

        for b0, b1 in tqdm(list(zip(batch_starts, batch_ends)), desc=f"Inference {split}"):
            events_x = []
            events_mask = []
            events_true = []

            for i in range(b0, b1):
                s, e = int(ev_starts[i]), int(ev_starts[i + 1])
                data = f[f"{split}/data/data"][s:e].astype(np.float32)
                labels = f[f"{split}/labels/data"][s:e]
                t_res = f[f"{split}/t_res/data"][s:e].astype(np.float32)

                sig = labels != 0
                n_sig = int(sig.sum())
                if n_sig < MIN_SIGNAL_HITS:
                    continue

                x_sig = data[sig]
                tres_sig = t_res[sig]
                true_abs = np.abs(tres_sig)

                L = x_sig.shape[0]
                events_x.append(x_sig)
                events_true.append(true_abs)
                events_mask.append(np.ones(L, dtype=np.float32))

            if not events_x:
                continue

            max_len = max(x.shape[0] for x in events_x)
            B = len(events_x)
            xb = np.zeros((B, max_len, 5), dtype=np.float32)
            mb = np.zeros((B, max_len), dtype=np.float32)
            for j, (x, m) in enumerate(zip(events_x, events_mask)):
                L = x.shape[0]
                xb[j, :L] = x
                mb[j, :L] = m

            xt = torch.tensor(xb, device=DEVICE)
            mt = torch.tensor(mb, device=DEVICE, dtype=torch.bool)

            with torch.no_grad():
                out = model(xt, mt)

            out_np = out.cpu().numpy()
            for j, true_abs in enumerate(events_true):
                L = events_x[j].shape[0]
                pred_norm = out_np[j, :L, 0]
                pred_abs = pred_norm * TRES_STD + TRES_MEAN

                pred_all.append(pred_abs.astype(np.float64))
                true_all.append(np.asarray(true_abs, dtype=np.float64))

            del xt, mt, out

    if not pred_all:
        return np.array([]), np.array([])

    return np.concatenate(pred_all), np.concatenate(true_all)


def metrics_subset(pred: np.ndarray, true: np.ndarray) -> dict:
    err = pred - true
    ae = np.abs(err)
    out = {
        "N": len(pred),
        "MAE": float(np.mean(ae)),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "Median_AE": float(np.median(ae)),
        "Q68": float(np.percentile(ae, 68)),
        "Q90": float(np.percentile(ae, 90)),
        "Bias": float(np.mean(err)),
        "R2": float(r2_score(true, pred)) if len(np.unique(true)) > 1 else float("nan"),
    }
    return out


def plot_two_panel(
    pred: np.ndarray,
    true: np.ndarray,
    title: str,
    out_path: Path,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # Predicted vs True (hexbin), focus 0–15 on true
    m = true < TRUE_MAX_NS
    t = true[m]
    p = pred[m]
    hb = ax1.hexbin(
        t,
        p,
        gridsize=55,
        cmap="inferno",
        mincnt=1,
        extent=(0, 15, 0, max(15.0, float(p.max()) * 1.05)),
    )
    ax1.plot([0, 15], [0, 15], "c--", lw=1.2, alpha=0.85)
    ax1.set_xlabel(r"True $|\mathrm{t_{res}}|$ (ns)")
    ax1.set_ylabel(r"Predicted $|\mathrm{t_{res}}|$ (ns)")
    ax1.set_title("Predicted vs true")
    ax1.set_xlim(0, 15)
    ax1.set_aspect("auto")
    plt.colorbar(hb, ax=ax1, label="counts")

    # Error vs |true|: median and Q68 in bins
    bins = np.linspace(0, TRUE_MAX_NS, 16)
    centers = 0.5 * (bins[:-1] + bins[1:])
    med = []
    q68 = []
    counts = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (true >= lo) & (true < hi)
        counts.append(int(sel.sum()))
        if sel.sum() < 20:
            med.append(np.nan)
            q68.append(np.nan)
            continue
        ae = np.abs(pred[sel] - true[sel])
        med.append(float(np.median(ae)))
        q68.append(float(np.percentile(ae, 68)))

    ax2.plot(centers, med, "o-", label="Median |error|", color="C0")
    ax2.plot(centers, q68, "s-", label="Q68 |error|", color="C1")
    ax2.set_xlabel(r"True $|\mathrm{t_{res}}|$ bin center (ns)")
    ax2.set_ylabel("|error| (ns)")
    ax2.set_title("Error vs true magnitude")
    ax2.legend()
    ax2.set_xlim(0, 15)
    ax2.grid(True)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, exp_dir, label in MODELS:
        ckpt = CKPT_BASE / exp_dir / "best_nue2.ckpt"
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)

        model = load_encoder(ckpt)
        pred, true = collect_predictions(model, split="val", max_events=30000)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        sel = true < TRUE_MAX_NS
        p = pred[sel]
        t = true[sel]
        m = metrics_subset(p, t)

        plot_two_panel(
            pred,
            true,
            title=f"{label} — nue2 val, GT signal hits, "
            rf"$|\mathrm{{t_{{res}}}}|$ true $< {TRUE_MAX_NS:g}$ ns ($N={m['N']}$ hits)",
            out_path=OUT_DIR / f"tres_eval_{key}_nue2_lt15.png",
        )

        rows.append(
            {
                "model": label,
                "key": key,
                "N_hits_lt15": m["N"],
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "Median_AE": m["Median_AE"],
                "Q68": m["Q68"],
                "Q90": m["Q90"],
                "Bias": m["Bias"],
                "R2": m["R2"],
            }
        )

    csv_path = OUT_DIR / "metrics_nue2_lt15.csv"
    with open(csv_path, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Markdown table for convenience
    md_path = OUT_DIR / "metrics_nue2_lt15.md"
    cols = [
        "model",
        "N_hits_lt15",
        "MAE",
        "RMSE",
        "Median_AE",
        "Q68",
        "Q90",
        "Bias",
        "R2",
    ]
    with open(md_path, "w") as fo:
        fo.write("| " + " | ".join(cols) + " |\n")
        fo.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for r in rows:
            fo.write(
                "| "
                + " | ".join(
                    [
                        str(r[c])
                        if c == "model"
                        else (f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]))
                        for c in cols
                    ]
                )
                + " |\n"
            )

    print(f"Wrote plots under {OUT_DIR}")
    print(f"Wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
