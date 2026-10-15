"""Paired bootstrap of domain-probe AUC and fixed-cohort post-cut W1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from da_stronger_train import (
    EXP_DEFAULT,
    MC_DEFAULT,
    accepted,
    infer,
    load_events,
    model_from_checkpoint,
    normalization,
)


def probe_probabilities(mc_train, exp_train, mc_test, exp_test):
    xtr = np.concatenate([mc_train["features"], exp_train["features"]])
    ytr = np.r_[
        np.zeros(len(mc_train["features"])), np.ones(len(exp_train["features"]))
    ]
    xte = np.concatenate([mc_test["features"], exp_test["features"]])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    clf.fit(xtr, ytr)
    return clf.predict_proba(xte)[:, 1]


def w1_on_events(mc_scores, exp_scores, mc_index, exp_index):
    return float(
        wasserstein_distance(
            np.concatenate([mc_scores[i] for i in mc_index]),
            np.concatenate([exp_scores[i] for i in exp_index]),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--selected", required=True)
    parser.add_argument("--base-proper-mask", action="store_true")
    parser.add_argument("--base-masked-pool", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mc", default=MC_DEFAULT)
    parser.add_argument("--exp", default=EXP_DEFAULT)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--bootstraps", type=int, default=150)
    args = parser.parse_args()
    torch.set_num_threads(4)
    target = normalization(args.mc)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    events = {
        "mc_probe": load_events(args.mc, "val", args.n, target, True),
        "exp_probe": load_events(args.exp, "val", args.n, target, False),
        "mc_first": load_events(args.mc, "val", args.n, target, True, args.n),
        "exp_first": load_events(args.exp, "test", args.n, target, False, 0),
        "mc_second": load_events(args.mc, "val", args.n, target, True, 2 * args.n),
        "exp_second": load_events(args.exp, "test", args.n, target, False, args.n),
    }
    print("events", {k: len(v) for k, v in events.items()}, flush=True)
    predictions = {}
    for name, checkpoint, proper, masked in (
        ("base", args.base, args.base_proper_mask, args.base_masked_pool),
        ("selected", args.selected, True, True),
    ):
        model = model_from_checkpoint(checkpoint, proper, masked).to(device)
        predictions[name] = {k: infer(model, ev, device) for k, ev in events.items()}
        print("inferred", name, flush=True)
        del model
        torch.cuda.empty_cache()

    rng = np.random.default_rng(42)
    result = {}
    for half in ("first", "second"):
        mc_events, exp_events = events[f"mc_{half}"], events[f"exp_{half}"]
        per_model = {}
        for name in ("base", "selected"):
            p = predictions[name]
            score = probe_probabilities(
                p["mc_probe"], p["exp_probe"], p[f"mc_{half}"], p[f"exp_{half}"]
            )
            mc_scores, exp_scores = (
                p[f"mc_{half}"]["scores"],
                p[f"exp_{half}"]["scores"],
            )
            per_model[name] = {
                "probe_score": score,
                "mc_scores": mc_scores,
                "exp_scores": exp_scores,
                "mc_cut": accepted(mc_events, mc_scores),
                "exp_cut": accepted(exp_events, exp_scores),
            }
        fixed_mc = np.flatnonzero(
            per_model["base"]["mc_cut"] & per_model["selected"]["mc_cut"]
        )
        fixed_exp = np.flatnonzero(
            per_model["base"]["exp_cut"] & per_model["selected"]["exp_cut"]
        )
        n = len(mc_events)
        y = np.r_[np.zeros(n), np.ones(len(exp_events))]
        metrics = {
            "n_fixed_mc": len(fixed_mc),
            "n_fixed_exp": len(fixed_exp),
            "probe_auc": {},
            "w1_fixed": {},
        }
        for name in ("base", "selected"):
            item = per_model[name]
            metrics["probe_auc"][name] = float(roc_auc_score(y, item["probe_score"]))
            metrics["w1_fixed"][name] = w1_on_events(
                item["mc_scores"], item["exp_scores"], fixed_mc, fixed_exp
            )
        auc_deltas, w1_deltas = [], []
        for _ in range(args.bootstraps):
            im = rng.integers(0, n, size=n)
            ie = rng.integers(0, len(exp_events), size=len(exp_events))
            ib = np.r_[im, n + ie]
            auc_deltas.append(
                roc_auc_score(y[ib], per_model["selected"]["probe_score"][ib])
                - roc_auc_score(y[ib], per_model["base"]["probe_score"][ib])
            )
            fm = rng.choice(fixed_mc, size=len(fixed_mc), replace=True)
            fe = rng.choice(fixed_exp, size=len(fixed_exp), replace=True)
            w1_deltas.append(
                w1_on_events(
                    per_model["selected"]["mc_scores"],
                    per_model["selected"]["exp_scores"],
                    fm,
                    fe,
                )
                - w1_on_events(
                    per_model["base"]["mc_scores"],
                    per_model["base"]["exp_scores"],
                    fm,
                    fe,
                )
            )
        metrics["probe_auc_delta_selected_minus_base_ci95"] = np.quantile(
            auc_deltas, [0.025, 0.975]
        ).tolist()
        metrics["w1_fixed_delta_selected_minus_base_ci95"] = np.quantile(
            w1_deltas, [0.025, 0.975]
        ).tolist()
        result[half] = metrics
        print("HALF", half, json.dumps(metrics), flush=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
