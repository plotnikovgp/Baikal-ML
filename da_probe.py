"""Frozen-encoder MC/experimental domain probes for noise-signal DA checkpoints.

Run on cluster63 with the Baikal-ML checkout.  Probe fitting never updates the
encoder and uses different event IDs for fitting and reporting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO / "pytorch_training"))
from models.encoder import EncoderDomainAdaptation  # noqa: E402

BASE = Path("/home/plotnikovgp/baikal/Baikal-ML/pytorch_training")
SCORES = Path("/home/plotnikovgp/tmp/angle_reconstruction/data")
PARTICLES = ("muatm", "nuatm", "nue2")
EXP = BASE / "data/baikal_exp_filtered_norm_w_model_enc_v2_fix_q.h5"
SRC = BASE / "data/baikal_2020_sig-noise_mid-eq_normed.h5"
CKPT_DIR = BASE / "checkpoints/noise_sig_experiments"
CHECKPOINTS = (
    "g_original_labels_da_hs128",
    "k_nsol_labelneq0_da_hs128_k0p0001",
    "k_nsol_labelneq0_da_hs128_k0p001",
    "k_nsol_labelneq0_da_hs128",
)


def norm(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path) as f:
        return (
            f["norm_param/mean"][()].astype(np.float32),
            f["norm_param/std"][()].astype(np.float32),
        )


def load_events(
    path: Path, split: str, indices: np.ndarray, target_norm: tuple, source: str
) -> dict:
    src_mean, src_std = norm(path)
    dst_mean, dst_std = target_norm
    out_x, out_mask, out_label, out_ids, out_nhits, out_nstrings = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    with h5py.File(path) as f:
        g = f[split]
        starts = g["ev_starts/data"][()]
        ids = g["ev_ids/data"][()] if "ev_ids" in g else np.arange(len(starts) - 1)
        nstrings = g["num_un_strings/data"][()] if "num_un_strings" in g else None
        for ix in indices:
            lo, hi = int(starts[ix]), int(starts[ix + 1])
            if hi - lo < 8 or (nstrings is not None and nstrings[ix] < 2):
                continue
            raw = g["data/data"][lo : min(hi, lo + 128)].astype(np.float32)
            x = (raw * src_std + src_mean - dst_mean) / dst_std
            mask = np.zeros(128, dtype=bool)
            mask[: len(x)] = True
            xpad = np.zeros((128, 5), dtype=np.float32)
            xpad[: len(x)] = x
            out_x.append(xpad)
            out_mask.append(mask)
            if source == "mc":
                out_label.append(
                    (g["labels/data"][lo : lo + len(x)] != 0).astype(np.uint8)
                )
            out_ids.append(str(ids[ix]))
            out_nhits.append(hi - lo)
            out_nstrings.append(int(nstrings[ix]) if nstrings is not None else -1)
    return {
        "x": np.stack(out_x),
        "mask": np.stack(out_mask),
        "labels": out_label,
        "ids": np.asarray(out_ids),
        "nhits": np.asarray(out_nhits),
        "nstrings": np.asarray(out_nstrings),
        "source": source,
    }


def extract(
    model, sample: dict, batch_size: int, device: str, masked_head: bool = False
) -> dict:
    exact, masked, head, hit_truth, hit_score = [], [], [], [], []
    x_all, m_all = sample["x"], sample["mask"]
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x_all), batch_size):
            x = torch.from_numpy(x_all[i : i + batch_size]).to(device)
            mask = torch.from_numpy(m_all[i : i + batch_size]).to(device)
            hit_logits, domain_logits, h = model(x, mask, return_hidden_states=True)
            exact.append(h.mean(1).float().cpu().numpy())
            masked.append(
                ((h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True))
                .float()
                .cpu()
                .numpy()
            )
            if masked_head:
                domain_logits = model.domain_classifier(
                    (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
                )
            head.append(torch.softmax(domain_logits, -1)[:, 1].float().cpu().numpy())
            if sample["source"] == "mc":
                prob = torch.softmax(hit_logits, -1)[:, :, 1].float().cpu().numpy()
                for j, labels in enumerate(sample["labels"][i : i + batch_size]):
                    hit_truth.extend(labels.tolist())
                    hit_score.extend(prob[j, : len(labels)].tolist())
    return {
        "exact": np.concatenate(exact),
        "masked": np.concatenate(masked),
        "head": np.concatenate(head),
        "hit_truth": np.asarray(hit_truth),
        "hit_score": np.asarray(hit_score),
    }


def auc(y, score):
    val = roc_auc_score(y, score)
    return {"auc": float(val), "separability_auc": float(max(val, 1 - val))}


def domain_metrics(a_train, b_train, a_test, b_test, key):
    x_train = np.concatenate([a_train[key], b_train[key]])
    y_train = np.r_[np.zeros(len(a_train[key])), np.ones(len(b_train[key]))]
    x_test = np.concatenate([a_test[key], b_test[key]])
    y_test = np.r_[np.zeros(len(a_test[key])), np.ones(len(b_test[key]))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
    clf.fit(x_train, y_train)
    return auc(y_test, clf.predict_proba(x_test)[:, 1])


def evaluate(model, raw, batch_size, device, masked_head=False):
    features = {
        k: extract(model, v, batch_size, device, masked_head) for k, v in raw.items()
    }
    result = {}
    for particle in (*PARTICLES, "mixed"):
        if particle == "mixed":
            mc_train = {
                k: np.concatenate([features[f"{p}_train"][k] for p in PARTICLES])
                for k in ("exact", "masked", "head")
            }
            mc_test = {
                k: np.concatenate([features[f"{p}_test"][k] for p in PARTICLES])
                for k in ("exact", "masked", "head")
            }
        else:
            mc_train = features[f"{particle}_train"]
            mc_test = features[f"{particle}_test"]
        n = len(mc_test["head"])
        exp_train = {
            k: features["exp_train"][k][: len(mc_train["head"])]
            for k in ("exact", "masked", "head")
        }
        exp_test = {k: features["exp_test"][k][:n] for k in ("exact", "masked", "head")}
        head_y = np.r_[np.zeros(n), np.ones(n)]
        head_score = np.r_[mc_test["head"], exp_test["head"]]
        result[particle] = {
            "n_per_domain_test": n,
            "head": auc(head_y, head_score),
            "probe_exact": domain_metrics(
                mc_train, exp_train, mc_test, exp_test, "exact"
            ),
            "probe_masked": domain_metrics(
                mc_train, exp_train, mc_test, exp_test, "masked"
            ),
        }
    yt = np.concatenate([features[f"{p}_test"]["hit_truth"] for p in PARTICLES])
    ys = np.concatenate([features[f"{p}_test"]["hit_score"] for p in PARTICLES])
    result["hit_task"] = {
        "roc_auc": float(roc_auc_score(yt, ys)),
        "average_precision": float(average_precision_score(yt, ys)),
        "precision_at_0p5": float(precision_score(yt, ys >= 0.5)),
        "recall_at_0p5": float(recall_score(yt, ys >= 0.5)),
        "n_hits": len(yt),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n", type=int, default=600, help="events per MC particle per probe split"
    )
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(41324)
    target_norm = norm(SRC)
    raw = {}
    for p in PARTICLES:
        path = SCORES / f"2020_{p}_scores_allhits_spaced20k.h5"
        choice = rng.permutation(20_000)[: 2 * args.n]
        raw[f"{p}_train"] = load_events(
            path, "test", np.sort(choice[: args.n]), target_norm, "mc"
        )
        raw[f"{p}_test"] = load_events(
            path, "test", np.sort(choice[args.n :]), target_norm, "mc"
        )
        assert not set(raw[f"{p}_train"]["ids"]) & set(raw[f"{p}_test"]["ids"])
    for split, key in (("val", "exp_train"), ("test", "exp_test")):
        with h5py.File(EXP) as f:
            amount = len(f[f"{split}/ev_starts/data"]) - 1
        choice = np.sort(rng.choice(amount, 3 * args.n, replace=False))
        raw[key] = load_events(EXP, split, choice, target_norm, "exp")
    assert not set(raw["exp_train"]["ids"]) & set(raw["exp_test"]["ids"])
    print("sample sizes", {k: len(v["x"]) for k, v in raw.items()}, flush=True)
    result = {
        "protocol": {
            "mc": "2020 test held-out, disjoint probe subsets",
            "exp": "val for probe fitting, test for probe reporting",
            "input": "original raw hits, first 128, fixed 128 padding; raw 2s8h cut",
            "n_requested_per_mc_particle_split": args.n,
        },
        "models": {},
    }
    for ck in CHECKPOINTS:
        folder = CKPT_DIR / ck
        with open(folder / "train_config_mc_2020.yaml") as f:
            conf = yaml.safe_load(f)
        model = EncoderDomainAdaptation(**conf["model"]["params"])
        state = torch.load(
            folder / "best_mc_2020.ckpt", map_location="cpu", weights_only=False
        )
        model.load_state_dict(state, strict=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print("evaluating", ck, "device", device, flush=True)
        result["models"][ck] = {
            "da_weight": conf["train_type"]["domain_adaptation_loss_k"],
            "lr": conf["training"]["lr"],
            "metrics": evaluate(model, raw, args.batch_size, device),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(ck, json.dumps(result["models"][ck]["metrics"]["mixed"]), flush=True)


if __name__ == "__main__":
    main()
