"""INLP ablation: force a frozen hit encoder's *linear* domain probe toward 0.5.

Projection directions are learned on train data. A separate validation split
selects the projection rank; the final MC/EXP test cohort is untouched until
that selection. The same projection is applied to every hit before the hit
head, so this is not merely post-processing the diagnostic event embedding.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from da_stronger_train import (
    EXP_DEFAULT,
    MC_DEFAULT,
    accepted,
    load_events,
    make_batch,
    model_from_checkpoint,
    normalization,
    pool_hidden,
)


@torch.no_grad()
def pooled(model, events, device, batch_size=32):
    out = []
    model.eval()
    for start in range(0, len(events), batch_size):
        x, mask, _ = make_batch(events[start : start + batch_size], device)
        _, hidden = model.encoder(x, mask)
        out.append(pool_hidden(hidden, mask).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def linear_probe(train_mc, train_exp, test_mc, test_exp):
    xtr = np.concatenate((train_mc, train_exp))
    ytr = np.r_[np.zeros(len(train_mc)), np.ones(len(train_exp))]
    xte = np.concatenate((test_mc, test_exp))
    yte = np.r_[np.zeros(len(test_mc)), np.ones(len(test_exp))]
    probe = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, solver="liblinear", max_iter=300)
    )
    probe.fit(xtr, ytr)
    auc = float(roc_auc_score(yte, probe.predict_proba(xte)[:, 1]))
    return auc, probe


def project(x, basis):
    if basis.shape[1] == 0:
        return x
    if basis.shape[1] >= x.shape[1]:
        return np.zeros_like(x)
    return x - (x @ basis) @ basis.T


def nonlinear_probe(train_mc, train_exp, test_mc, test_exp):
    xtr = np.concatenate((train_mc, train_exp))
    ytr = np.r_[np.zeros(len(train_mc)), np.ones(len(train_exp))]
    xte = np.concatenate((test_mc, test_exp))
    yte = np.r_[np.zeros(len(test_mc)), np.ones(len(test_exp))]
    probe = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=42,
        n_jobs=4,
    )
    probe.fit(xtr, ytr)
    return float(roc_auc_score(yte, probe.predict_proba(xte)[:, 1]))


@torch.no_grad()
def hit_scores(model, head, events, basis, device, batch_size=32):
    scores = []
    q = torch.as_tensor(basis, dtype=torch.float32, device=device)
    model.eval()
    head.eval()
    for start in range(0, len(events), batch_size):
        group = events[start : start + batch_size]
        x, mask, _ = make_batch(group, device)
        _, hidden = model.encoder(x, mask)
        if basis.shape[1] >= hidden.shape[-1]:
            hidden = torch.zeros_like(hidden)
        elif basis.shape[1]:
            hidden = hidden - (hidden @ q) @ q.T
        score = F.softmax(head(hidden), dim=-1)[:, :, 1].cpu().numpy()
        scores.extend(score[i, : len(e["x"])] for i, e in enumerate(group))
    return scores


@torch.no_grad()
def collapsed_scores(head, events, device):
    probability = float(F.softmax(head(torch.zeros(1, 128, device=device)), -1)[0, 1])
    return [np.full(len(e["x"]), probability, np.float32) for e in events]


def hit_metrics(mc_events, exp_events, mc_scores, exp_scores):
    truth = np.concatenate([e["y"] for e in mc_events])
    score = np.concatenate(mc_scores)
    mc_cut = accepted(mc_events, mc_scores)
    exp_cut = accepted(exp_events, exp_scores)
    result = {
        "hit_ap": float(average_precision_score(truth, score)),
        "hit_auc": float(roc_auc_score(truth, score)),
        "precision_p0p5": float(
            np.sum((score >= 0.5) & (truth == 1)) / max(1, np.sum(score >= 0.5))
        ),
        "recall_p0p5": float(
            np.sum((score >= 0.5) & (truth == 1)) / max(1, np.sum(truth == 1))
        ),
        "n_mc_pass": int(mc_cut.sum()),
        "n_exp_pass": int(exp_cut.sum()),
        "w1_all_hits": float(wasserstein_distance(score, np.concatenate(exp_scores))),
    }
    if mc_cut.any() and exp_cut.any():
        result["w1_predicted_2s8h"] = float(
            wasserstein_distance(
                np.concatenate([s for s, keep in zip(mc_scores, mc_cut) if keep]),
                np.concatenate([s for s, keep in zip(exp_scores, exp_cut) if keep]),
            )
        )
    return result


def train_projected_head(model, basis, mc_train, device, steps=400, batch_size=32):
    head = copy.deepcopy(model.main_head).to(device)
    head.train()
    q = torch.as_tensor(basis, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    rng = np.random.default_rng(42)
    losses = []
    for step in range(1, steps + 1):
        indices = rng.integers(0, len(mc_train), size=batch_size)
        x, mask, truth = make_batch([mc_train[int(i)] for i in indices], device)
        with torch.no_grad():
            _, hidden = model.encoder(x, mask)
            if basis.shape[1] >= hidden.shape[-1]:
                hidden = torch.zeros_like(hidden)
            else:
                hidden = hidden - (hidden @ q) @ q.T
        logits = head(hidden)
        loss = F.cross_entropy(logits[mask], truth[mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            losses.append(float(loss.detach()))
            print("HEAD_TRAIN", step, losses[-1], flush=True)
    return head, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mc", default=MC_DEFAULT)
    parser.add_argument("--exp", default=EXP_DEFAULT)
    parser.add_argument("--max-directions", type=int, default=128)
    parser.add_argument("--head-steps", type=int, default=400)
    parser.add_argument(
        "--projection-source",
        choices=("train", "val", "test_calibration"),
        default="train",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    threadpool_limits(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target_norm = normalization(args.mc)
    print("LOADING", flush=True)
    mc_train = load_events(args.mc, "train", 60000, target_norm, True)
    rng = np.random.default_rng(42)
    if args.projection_source == "train":
        mc_domain_all = [e for e in mc_train if e["id"].startswith(b"muatm_")]
        mc_domain = [
            mc_domain_all[int(i)]
            for i in rng.choice(len(mc_domain_all), 10000, replace=False)
        ]
        exp_train_all = load_events(args.exp, "train", 30000, target_norm, False)
        exp_domain = [
            exp_train_all[int(i)]
            for i in rng.choice(len(exp_train_all), 10000, replace=False)
        ]
        mc_val = load_events(args.mc, "val", 5000, target_norm, True)
        exp_val = load_events(args.exp, "val", 5000, target_norm, False)
    elif args.projection_source == "val":
        mc_val_all = load_events(args.mc, "val", 10000, target_norm, True)
        exp_val_all = load_events(args.exp, "val", 10000, target_norm, False)
        mc_domain, mc_val = mc_val_all[:5000], mc_val_all[5000:]
        exp_domain, exp_val = exp_val_all[:5000], exp_val_all[5000:]
    else:
        mc_val_all = load_events(args.mc, "val", 10000, target_norm, True)
        mc_domain, mc_val = mc_val_all[:5000], mc_val_all[5000:]
        exp_domain = load_events(args.exp, "val", 5000, target_norm, False)
        exp_val = load_events(args.exp, "test", 5000, target_norm, False)
    mc_test = load_events(args.mc, "val", 5000, target_norm, True, 10000)
    exp_test = load_events(args.exp, "test", 5000, target_norm, False, 5000)
    print(
        "COUNTS",
        {
            "mc_train": len(mc_train),
            "mc_domain": len(mc_domain),
            "exp_domain": len(exp_domain),
            "mc_val": len(mc_val),
            "exp_val": len(exp_val),
            "mc_test": len(mc_test),
            "exp_test": len(exp_test),
        },
        flush=True,
    )
    model = model_from_checkpoint(args.base, True, True).to(device).eval()
    features = {}
    for name, events in (
        ("mc_domain", mc_domain),
        ("exp_domain", exp_domain),
        ("mc_val", mc_val),
        ("exp_val", exp_val),
        ("mc_test", mc_test),
        ("exp_test", exp_test),
    ):
        features[name] = pooled(model, events, device)
        print("FEATURES", name, features[name].shape, flush=True)

    x_domain = np.concatenate((features["mc_domain"], features["exp_domain"]))
    x_val = np.concatenate((features["mc_val"], features["exp_val"]))
    y_domain = np.r_[np.zeros(len(mc_domain)), np.ones(len(exp_domain))]
    x_domain_work = x_domain.astype(np.float64)
    x_val_work = x_val.astype(np.float64)
    basis = np.empty((128, 0), np.float64)
    history = []
    milestones = {0: "baseline"}
    thresholds = (0.8, 0.7, 0.6, 0.55, 0.52, 0.51, 0.505)
    crossed_thresholds = set()

    def validation_auc():
        auc, _ = linear_probe(
            x_val_work[:2500],
            x_val_work[5000:7500],
            x_val_work[2500:5000],
            x_val_work[7500:10000],
        )
        return max(auc, 1.0 - auc)

    auc = validation_auc()
    history.append({"rank_removed": 0, "validation_probe_auc_symmetric": auc})
    print("PROJECTION", 0, auc, flush=True)
    for k in range(1, args.max_directions + 1):
        removal_probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, solver="liblinear", max_iter=300),
        )
        removal_probe.fit(x_domain_work, y_domain)
        scale = removal_probe.named_steps["standardscaler"].scale_
        direction = removal_probe.named_steps["logisticregression"].coef_[0] / scale
        original_norm = float(np.linalg.norm(direction))
        if basis.shape[1]:
            for _ in range(2):
                direction -= basis @ (basis.T @ direction)
        norm = float(np.linalg.norm(direction))
        if norm < max(1e-12, 1e-6 * original_norm):
            print("STOP_LOW_NORM", k, norm, original_norm, flush=True)
            break
        direction = direction / norm
        basis = np.column_stack((basis, direction))
        x_domain_work -= np.outer(x_domain_work @ direction, direction)
        x_val_work -= np.outer(x_val_work @ direction, direction)
        if basis.shape[1] >= x_val_work.shape[1]:
            x_domain_work.fill(0)
            x_val_work.fill(0)
            auc = 0.5
        else:
            auc = validation_auc()
        history.append({"rank_removed": k, "validation_probe_auc_symmetric": auc})
        for threshold in thresholds:
            if auc <= threshold and threshold not in crossed_thresholds:
                milestones[k] = f"validation_below_{threshold}"
                crossed_thresholds.add(threshold)
        print("PROJECTION", k, auc, flush=True)
        if auc <= 0.51:
            break

    # Keep milestone ranks and the last rank, plus a true constant-feature control.
    fixed_ranks = {0, 4, 16, 32, 64, 80, 88, 90, 92, 96, 100, 108, 116, 120}
    ranks = sorted(
        (set(milestones) | fixed_ranks | {basis.shape[1]})
        & set(range(basis.shape[1] + 1))
    )
    results = {}
    for k in ranks:
        q = basis[:, :k]
        val_mc = project(features["mc_val"], q)
        val_exp = project(features["exp_val"], q)
        test_mc = project(features["mc_test"], q)
        test_exp = project(features["exp_test"], q)
        probe_auc, _ = linear_probe(val_mc, val_exp, test_mc, test_exp)
        mc_scores = hit_scores(model, model.main_head, mc_test, q, device)
        exp_scores = hit_scores(model, model.main_head, exp_test, q, device)
        metrics = hit_metrics(mc_test, exp_test, mc_scores, exp_scores)
        metrics["probe_auc"] = probe_auc
        metrics["validation_probe_auc_symmetric"] = history[k][
            "validation_probe_auc_symmetric"
        ]
        metrics["milestone"] = milestones.get(k, "last")
        results[str(k)] = metrics
        print("RESULT", k, json.dumps(metrics), flush=True)

    # Nonlinear challenge to the first rank selected by the validation probe.
    chosen = next(
        (k for k in ranks if history[k]["validation_probe_auc_symmetric"] <= 0.51),
        basis.shape[1],
    )
    q = basis[:, :chosen]
    nonlinear_auc = nonlinear_probe(
        project(features["mc_val"], q),
        project(features["exp_val"], q),
        project(features["mc_test"], q),
        project(features["exp_test"], q),
    )
    print("NONLINEAR_PROBE", chosen, nonlinear_auc, flush=True)

    if chosen > 0:
        head, losses = train_projected_head(model, q, mc_train, device, args.head_steps)
        mc_scores = hit_scores(model, head, mc_test, q, device)
        exp_scores = hit_scores(model, head, exp_test, q, device)
        recovered = hit_metrics(mc_test, exp_test, mc_scores, exp_scores)
        print("RECOVERED_HEAD", chosen, json.dumps(recovered), flush=True)
        torch.save(
            {
                "base": args.base,
                "projection_basis": q,
                "head_state_dict": head.state_dict(),
                "proper_mask": True,
                "masked_pool": True,
            },
            args.out / "projected_head.ckpt",
        )
    else:
        losses, recovered = [], None

    constant_mc = collapsed_scores(model.main_head, mc_test, device)
    constant_exp = collapsed_scores(model.main_head, exp_test, device)
    collapsed = hit_metrics(mc_test, exp_test, constant_mc, constant_exp)
    collapsed["probe_auc"] = 0.5
    print("FULL_COLLAPSE", json.dumps(collapsed), flush=True)
    summary = {
        "protocol": {
            "projection_train_per_domain": len(mc_domain),
            "projection_source": args.projection_source,
            "validation_probe_train_per_domain": 2500,
            "validation_probe_test_per_domain": 2500,
            "independent_holdout_per_domain": 5000,
            "selection_rule": "first validation symmetric AUC <= 0.51",
        },
        "projection_history": history,
        "results": results,
        "chosen_rank": int(chosen),
        "chosen_nonlinear_probe_auc": nonlinear_auc,
        "head_recovery_losses": losses,
        "head_recovery_metrics": recovered,
        "full_collapse": collapsed,
    }
    (args.out / "results.json").write_text(json.dumps(summary, indent=2))
    np.save(args.out / "projection_basis.npy", basis)


if __name__ == "__main__":
    main()
