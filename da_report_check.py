"""Audit report-era signal/noise DA vs no-DA on identical held-out events.

Unlike generate_report.py, reports both the legacy sigmoid(logit_1) score and
the correct two-class softmax score.  A frozen-encoder logreg probe is fitted
on MC val / EXP val and tested on disjoint MC val / EXP test events.
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
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pytorch_training"))
from models.encoder import Encoder, EncoderDomainAdaptation  # noqa: E402

BASE = Path("/home/plotnikovgp/baikal/Baikal-ML/pytorch_training")
MC_PATH = Path(
    "/home/plotnikovgp/baikal/data/baikal_2020_sig-noise_mid-eq_normed_5gb.h5"
)
EXP_PATH = BASE / "data/baikal_exp_filtered_norm_w_model_enc_v2_fix_q.h5"
NO_DA_DIR = BASE / "checkpoints/noise_sig_experiments/k_nsol_labelneq0_hs128"
DA_DIR = BASE / "checkpoints/noise_sig_experiments/k_nsol_labelneq0_da_hs128_k0p0001"


def norm(path):
    with h5py.File(path) as f:
        return f["norm_param/mean"][:].astype(np.float32), f["norm_param/std"][
            :
        ].astype(np.float32)


def select_mc_indices(n_each):
    with h5py.File(MC_PATH) as f:
        ids = f["val/ev_ids/data"][:]
    candidates = np.asarray(
        [i for i, eid in enumerate(ids) if eid.startswith(b"muatm_")]
    )
    if len(candidates) < 2 * n_each:
        raise ValueError(f"only {len(candidates)} muatm val events")
    return candidates[:n_each], candidates[n_each : 2 * n_each]


def load(path, split, indices, target_norm, is_mc):
    src_mean, src_std = norm(path)
    dst_mean, dst_std = target_norm
    events = []
    with h5py.File(path) as f:
        g = f[split]
        starts = g["ev_starts/data"][:]
        ids = g["ev_ids/data"][:] if "ev_ids" in g else None
        for ix in indices:
            lo, hi = int(starts[ix]), int(starts[ix + 1])
            if hi - lo < 8:
                continue
            channels = g["channels/data"][lo:hi]
            if len(np.unique(channels // 36)) < 2:
                continue
            raw = g["data/data"][lo : min(hi, lo + 256)].astype(np.float32)
            x = (raw * src_std + src_mean - dst_mean) / dst_std
            event = {
                "x": x,
                "channels": channels[: len(x)].astype(np.int32),
                "id": str(ids[ix]) if ids is not None else str(ix),
            }
            if is_mc:
                event["truth"] = (g["labels/data"][lo : lo + len(x)] != 0).astype(
                    np.uint8
                )
            events.append(event)
    return events


def load_model(da):
    folder = DA_DIR if da else NO_DA_DIR
    with open(
        folder / ("train_config_mc_2020.yaml" if da else "train_config.yaml")
    ) as f:
        conf = yaml.safe_load(f)
    model = (
        EncoderDomainAdaptation(**conf["model"]["params"])
        if da
        else Encoder(**conf["model"]["params"], return_hidden=True)
    )
    state = torch.load(
        folder / ("best_mc_2020.ckpt" if da else "best.ckpt"),
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(state, strict=True)
    return model.eval()


def infer(model, events, batch_size, device, da):
    model.to(device)
    features, domain_head, soft_scores, legacy_scores = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(events), batch_size):
            group = events[start : start + batch_size]
            max_len = max(len(ev["x"]) for ev in group)
            x = np.zeros((len(group), max_len, 5), dtype=np.float32)
            mask = np.zeros((len(group), max_len), dtype=bool)
            for i, ev in enumerate(group):
                n = len(ev["x"])
                x[i, :n] = ev["x"]
                mask[i, :n] = True
            xt = torch.from_numpy(x).to(device)
            mt = torch.from_numpy(mask).to(device)
            if da:
                logits, dom_logits, h = model(xt, mt, return_hidden_states=True)
                domain_head.extend(
                    torch.softmax(dom_logits, -1)[:, 1].cpu().numpy().tolist()
                )
            else:
                logits, h = model(xt, mt)
            pooled = (h * mt.unsqueeze(-1)).sum(1) / mt.sum(1, keepdim=True)
            features.append(pooled.float().cpu().numpy())
            probs_soft = torch.softmax(logits, -1)[:, :, 1].float().cpu().numpy()
            probs_legacy = torch.sigmoid(logits[:, :, 1]).float().cpu().numpy()
            for i, ev in enumerate(group):
                n = len(ev["x"])
                soft_scores.append(probs_soft[i, :n])
                legacy_scores.append(probs_legacy[i, :n])
    return {
        "features": np.concatenate(features),
        "head": np.asarray(domain_head),
        "softmax": soft_scores,
        "legacy_sigmoid": legacy_scores,
    }


def probe(train_mc, train_exp, test_mc, test_exp):
    xtr = np.concatenate([train_mc["features"], train_exp["features"]])
    ytr = np.r_[
        np.zeros(len(train_mc["features"])), np.ones(len(train_exp["features"]))
    ]
    xte = np.concatenate([test_mc["features"], test_exp["features"]])
    yte = np.r_[np.zeros(len(test_mc["features"])), np.ones(len(test_exp["features"]))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    clf.fit(xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(xte)[:, 1]))


def pass_cut(scores, event, threshold=0.5):
    keep = scores > threshold
    return int(keep.sum()) >= 8 and len(np.unique(event["channels"][keep] // 36)) >= 2


def score_metrics(model_output, mc_events, exp_events, score_key, fixed_cohort=None):
    m_scores, e_scores = (
        model_output["mc_test"][score_key],
        model_output["exp_test"][score_key],
    )
    yt = np.concatenate([ev["truth"] for ev in mc_events])
    ys = np.concatenate(m_scores)
    result = {
        "mc_hit_auc": float(roc_auc_score(yt, ys)),
        "mc_hit_ap": float(average_precision_score(yt, ys)),
        "mc_hit_precision_p0p5": float(precision_score(yt, ys >= 0.5)),
        "mc_hit_recall_p0p5": float(recall_score(yt, ys >= 0.5)),
        "w1_all_hits": float(
            wasserstein_distance(np.concatenate(m_scores), np.concatenate(e_scores))
        ),
    }
    m_cut = np.array([pass_cut(s, ev) for s, ev in zip(m_scores, mc_events)])
    e_cut = np.array([pass_cut(s, ev) for s, ev in zip(e_scores, exp_events)])
    result["n_mc_pass"] = int(m_cut.sum())
    result["n_exp_pass"] = int(e_cut.sum())
    if m_cut.any() and e_cut.any():
        result["w1_model_specific_cut"] = float(
            wasserstein_distance(
                np.concatenate([s for s, k in zip(m_scores, m_cut) if k]),
                np.concatenate([s for s, k in zip(e_scores, e_cut) if k]),
            )
        )
    if fixed_cohort is not None:
        fm, fe = fixed_cohort
        if fm.any() and fe.any():
            result["w1_same_events"] = float(
                wasserstein_distance(
                    np.concatenate([s for s, k in zip(m_scores, fm) if k]),
                    np.concatenate([s for s, k in zip(e_scores, fe) if k]),
                )
            )
            result["n_mc_same_events"] = int(fm.sum())
            result["n_exp_same_events"] = int(fe.sum())
    return result, (m_cut, e_cut)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    target_norm = norm(MC_PATH)
    mc_train_ix, mc_test_ix = select_mc_indices(args.n)
    raw = {
        "mc_train": load(MC_PATH, "val", mc_train_ix, target_norm, True),
        "mc_test": load(MC_PATH, "val", mc_test_ix, target_norm, True),
        "exp_train": load(EXP_PATH, "val", np.arange(args.n), target_norm, False),
        "exp_test": load(EXP_PATH, "test", np.arange(args.n), target_norm, False),
    }
    assert not set(ev["id"] for ev in raw["mc_train"]) & set(
        ev["id"] for ev in raw["mc_test"]
    )
    assert not set(ev["id"] for ev in raw["exp_train"]) & set(
        ev["id"] for ev in raw["exp_test"]
    )
    print("events", {k: len(v) for k, v in raw.items()}, flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = {}
    for name, is_da in (("no_da", False), ("da", True)):
        print("evaluating", name, flush=True)
        model = load_model(is_da)
        prediction = {
            key: infer(model, events, args.batch_size, device, is_da)
            for key, events in raw.items()
        }
        output[name] = prediction
        del model
        torch.cuda.empty_cache()
    result = {
        "protocol": {
            "mc": "2020 5GB val, disjoint muatm events",
            "exp": "val probe-train, test probe-test",
            "cut": "raw 2s8h for all; predicted 2s8h separately",
            "n_requested_per_domain_per_split": args.n,
        },
        "models": {},
    }
    for name in ("no_da", "da"):
        pred = output[name]
        one = {
            "probe_auc": probe(
                pred["mc_train"], pred["exp_train"], pred["mc_test"], pred["exp_test"]
            )
        }
        if name == "da":
            y = np.r_[np.zeros(len(raw["mc_test"])), np.ones(len(raw["exp_test"]))]
            score = np.r_[pred["mc_test"]["head"], pred["exp_test"]["head"]]
            one["domain_head_auc"] = float(roc_auc_score(y, score))
        result["models"][name] = one
    for score_key in ("softmax", "legacy_sigmoid"):
        measures, cuts = {}, {}
        for name in ("no_da", "da"):
            measures[name], cuts[name] = score_metrics(
                output[name], raw["mc_test"], raw["exp_test"], score_key
            )
        fixed = (cuts["no_da"][0] & cuts["da"][0], cuts["no_da"][1] & cuts["da"][1])
        for name in ("no_da", "da"):
            metrics, _ = score_metrics(
                output[name], raw["mc_test"], raw["exp_test"], score_key, fixed
            )
            result["models"][name][score_key] = metrics
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
