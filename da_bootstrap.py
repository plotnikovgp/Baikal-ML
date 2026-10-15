"""Paired bootstrap uncertainty for frozen-encoder domain-probe AUC differences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from da_probe import PARTICLES, extract
from da_sweep import INITIAL, build_data
from models.encoder import EncoderDomainAdaptation

PAIRS = (
    ("control", "da_masked_lr_only", False),
    ("paper_control", "paper_da", True),
    ("muatm_control", "muatm_da", True),
)


def features(name, probe, out_dir, model_params, device, batch_size):
    model = EncoderDomainAdaptation(**model_params)
    model.load_state_dict(
        torch.load(out_dir / f"{name}_e5.ckpt", map_location="cpu", weights_only=False),
        strict=True,
    )
    model.to(device)
    return {
        k: extract(model, v, batch_size, device)["masked"] for k, v in probe.items()
    }


def scores(feat, particle):
    if particle == "mixed":
        mc_train = np.concatenate([feat[f"{p}_train"] for p in PARTICLES])
        mc_test = np.concatenate([feat[f"{p}_test"] for p in PARTICLES])
    else:
        mc_train, mc_test = feat[f"{particle}_train"], feat[f"{particle}_test"]
    exp_train = feat["exp_train"][: len(mc_train)]
    exp_test = feat["exp_test"][: len(mc_test)]
    xtr = np.concatenate([mc_train, exp_train])
    ytr = np.r_[np.zeros(len(mc_train)), np.ones(len(exp_train))]
    xte = np.concatenate([mc_test, exp_test])
    yte = np.r_[np.zeros(len(mc_test)), np.ones(len(exp_test))]
    probe_model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, C=1.0)
    )
    probe_model.fit(xtr, ytr)
    return yte, probe_model.predict_proba(xte)[:, 1]


def bootstrap_pair(y, control_score, da_score, reps, rng):
    n = len(y) // 2
    assert len(y) == 2 * n and np.all(y[:n] == 0) and np.all(y[n:] == 1)
    base = roc_auc_score(y, control_score)
    da = roc_auc_score(y, da_score)
    diffs = []
    control_vals = []
    da_vals = []
    for _ in range(reps):
        ix = np.r_[rng.integers(0, n, n), n + rng.integers(0, n, n)]
        a = roc_auc_score(y[ix], control_score[ix])
        b = roc_auc_score(y[ix], da_score[ix])
        control_vals.append(a)
        da_vals.append(b)
        diffs.append(b - a)
    return {
        "n_per_domain": n,
        "control_auc": float(base),
        "da_auc": float(da),
        "auc_difference_da_minus_control": float(da - base),
        "control_ci95": np.quantile(control_vals, [0.025, 0.975]).tolist(),
        "da_ci95": np.quantile(da_vals, [0.025, 0.975]).tolist(),
        "difference_ci95": np.quantile(diffs, [0.025, 0.975]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(INITIAL / "train_config_mc_2020.yaml") as f:
        params = yaml.safe_load(f)["model"]["params"]
    result = {}
    rng = np.random.default_rng(7394)
    for control_name, da_name, paper_like in PAIRS:
        _, _, probe = build_data(600, 2500, paper_like)
        a = features(control_name, probe, args.output_dir, params, device, 96)
        b = features(da_name, probe, args.output_dir, params, device, 96)
        row = {}
        for particle in ("mixed", "muatm"):
            y, score_a = scores(a, particle)
            y_b, score_b = scores(b, particle)
            assert np.array_equal(y, y_b)
            row[particle] = bootstrap_pair(
                y, score_a, score_b, args.bootstrap_reps, rng
            )
        result[f"{control_name}_vs_{da_name}"] = row
        print(control_name, da_name, row, flush=True)
        args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
