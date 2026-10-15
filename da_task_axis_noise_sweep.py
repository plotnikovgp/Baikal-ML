"""Artificially erase residual domain signal in the task logit via event jitter.

The 128D embedding is first reduced to the hit-head logit-difference axis;
this discards all information the current hit head does not use. Gaussian
noise is then added to that logit difference, with one independent draw per
event shared by all its hits. Sigma is selected without the final holdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from da_force_invariance import hit_metrics
from da_stronger_train import (
    EXP_DEFAULT,
    MC_DEFAULT,
    load_events,
    make_batch,
    model_from_checkpoint,
    normalization,
)


@torch.no_grad()
def infer_delta(model, events, device):
    pooled, hits = [], []
    model.eval()
    for start in range(0, len(events), 32):
        group = events[start : start + 32]
        x, mask, _ = make_batch(group, device)
        logits, _ = model(x, mask)
        delta = (logits[:, :, 1] - logits[:, :, 0]).cpu().numpy()
        for i, event in enumerate(group):
            hit_delta = delta[i, : len(event["x"])].astype(np.float32)
            hits.append(hit_delta)
            pooled.append(float(hit_delta.mean()))
    return np.asarray(pooled, np.float32), hits


def logreg_auc(train_mc, train_exp, test_mc, test_exp):
    xtr = np.r_[train_mc, train_exp].reshape(-1, 1)
    ytr = np.r_[np.zeros(len(train_mc)), np.ones(len(train_exp))]
    xte = np.r_[test_mc, test_exp].reshape(-1, 1)
    yte = np.r_[np.zeros(len(test_mc)), np.ones(len(test_exp))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1, solver="liblinear"))
    clf.fit(xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(xte)[:, 1]))


def forest_auc(train_mc, train_exp, test_mc, test_exp):
    xtr = np.r_[train_mc, train_exp].reshape(-1, 1)
    ytr = np.r_[np.zeros(len(train_mc)), np.ones(len(train_exp))]
    xte = np.r_[test_mc, test_exp].reshape(-1, 1)
    yte = np.r_[np.zeros(len(test_mc)), np.ones(len(test_exp))]
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=4,
    )
    clf.fit(xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(xte)[:, 1]))


def scores_with_jitter(hit_delta, z, sigma):
    return [
        (1.0 / (1.0 + np.exp(-np.clip(d + sigma * offset, -40, 40)))).astype(np.float32)
        for d, offset in zip(hit_delta, z)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mc", default=MC_DEFAULT)
    parser.add_argument("--exp", default=EXP_DEFAULT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    threadpool_limits(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target = normalization(args.mc)
    events = {
        "mc_probe": load_events(args.mc, "val", 5000, target, True),
        "exp_probe": load_events(args.exp, "val", 5000, target, False),
        "mc_cal": load_events(args.mc, "val", 5000, target, True, 5000),
        "exp_cal": load_events(args.exp, "test", 5000, target, False),
        "mc_test": load_events(args.mc, "val", 5000, target, True, 10000),
        "exp_test": load_events(args.exp, "test", 5000, target, False, 5000),
    }
    model = model_from_checkpoint(args.base, True, True).to(device).eval()
    base = {}
    for name, ev in events.items():
        mean, hit = infer_delta(model, ev, device)
        base[name] = {"mean": mean, "hit": hit}
        print("INFERRED", name, len(ev), flush=True)

    sigmas = (0, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128)
    seeds = (41, 42, 43)
    result = {
        "protocol": {
            "noise": "one Gaussian logit offset per event",
            "probe_train_per_domain": 5000,
            "calibration_per_domain": 5000,
            "holdout_per_domain": 5000,
            "sigmas": sigmas,
            "seeds": seeds,
        },
        "sweep": {},
    }
    for sigma in sigmas:
        per_seed = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            jitter = {
                name: rng.normal(size=len(events[name])).astype(np.float32)
                for name in events
            }
            means = {name: base[name]["mean"] + sigma * jitter[name] for name in events}
            cal_auc = logreg_auc(
                means["mc_probe"],
                means["exp_probe"],
                means["mc_cal"],
                means["exp_cal"],
            )
            test_auc = logreg_auc(
                means["mc_probe"],
                means["exp_probe"],
                means["mc_test"],
                means["exp_test"],
            )
            mc_scores = scores_with_jitter(
                base["mc_test"]["hit"], jitter["mc_test"], sigma
            )
            exp_scores = scores_with_jitter(
                base["exp_test"]["hit"], jitter["exp_test"], sigma
            )
            quality = hit_metrics(
                events["mc_test"], events["exp_test"], mc_scores, exp_scores
            )
            per_seed.append(
                {
                    "calibration_probe_auc": cal_auc,
                    "calibration_probe_auc_symmetric": max(cal_auc, 1 - cal_auc),
                    "holdout_probe_auc": test_auc,
                    "holdout_probe_auc_symmetric": max(test_auc, 1 - test_auc),
                    **quality,
                }
            )
        summary = {}
        for key in (
            "calibration_probe_auc",
            "calibration_probe_auc_symmetric",
            "holdout_probe_auc",
            "holdout_probe_auc_symmetric",
            "hit_ap",
            "hit_auc",
            "precision_p0p5",
            "recall_p0p5",
            "n_mc_pass",
            "n_exp_pass",
            "w1_all_hits",
            "w1_predicted_2s8h",
        ):
            values = [item[key] for item in per_seed if key in item]
            if values:
                summary[key] = {
                    "mean": float(np.mean(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
        result["sweep"][str(sigma)] = {"summary": summary, "seeds": per_seed}
        print("SIGMA", sigma, json.dumps(summary), flush=True)

    # The nonlinear probe can still separate domains after the linear probe
    # approaches 0.5; report that explicitly for baseline and selected sigma.
    selected = next(
        (
            sigma
            for sigma in sigmas
            if result["sweep"][str(sigma)]["summary"][
                "calibration_probe_auc_symmetric"
            ]["mean"]
            <= 0.51
        ),
        sigmas[-1],
    )
    result["selected_sigma"] = selected
    rng = np.random.default_rng(42)
    jitter = {
        name: rng.normal(size=len(events[name])).astype(np.float32) for name in events
    }
    for sigma in (0, selected):
        means = {name: base[name]["mean"] + sigma * jitter[name] for name in events}
        result[f"forest_auc_sigma_{sigma}"] = forest_auc(
            means["mc_probe"],
            means["exp_probe"],
            means["mc_test"],
            means["exp_test"],
        )
        print("FOREST", sigma, result[f"forest_auc_sigma_{sigma}"], flush=True)
    (args.out / "results.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
