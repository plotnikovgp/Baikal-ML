"""Exact mean-pooled AUC=0.5 by event-centering the hit-task axis.

The hit head only uses one direction (difference of its two linear weights).
Keep that direction and subtract its event mean from every hit. An arbitrary
global logit offset is calibrated on labeled MC, never on the holdout. The
mean-pooled event representation is then constant, but a set-based probe may
still detect MC/EXP differences in the within-event hit distribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from da_force_invariance import hit_metrics
from da_stronger_train import (
    EXP_DEFAULT,
    MC_DEFAULT,
    EncoderDomainAdaptation,
    accepted,
    load_events,
    make_batch,
    model_from_checkpoint,
    normalization,
)
from da_task_axis_noise_sweep import infer_delta, scores_with_jitter


class CenteredHitAxisAdapter(nn.Module):
    """A deterministic 1D encoder output whose masked event mean is constant."""

    def __init__(self, base: nn.Module, global_logit_offset: float):
        super().__init__()
        self.base = base
        self.register_buffer("global_logit_offset", torch.tensor(global_logit_offset))

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        _, hidden = self.base.encoder(x, mask)
        head = self.base.main_head
        direction = head.weight[1] - head.weight[0]
        bias_difference = head.bias[1] - head.bias[0]
        original_delta = hidden @ direction + bias_difference
        weight = mask.to(original_delta.dtype)
        event_mean = (original_delta * weight).sum(1, keepdim=True) / weight.sum(
            1, keepdim=True
        ).clamp_min(1)
        centered_delta = original_delta - event_mean + self.global_logit_offset
        projected = ((centered_delta - bias_difference) / direction.square().sum())[
            ..., None
        ] * direction
        logits = head(projected)
        pooled = (projected * weight[..., None]).sum(1) / weight.sum(
            1, keepdim=True
        ).clamp_min(1)
        return logits, pooled


def load_adapter_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Load the self-contained base-network plus centering-adapter state dict."""
    base = EncoderDomainAdaptation(
        in_features=5,
        hidden_size=128,
        num_layers=5,
        dim_feedforward_size=512,
        n_heads=1,
        out_size=2,
        dropout_p=0.0,
        num_domains=2,
        domain_classifier_hidden_size=128,
        domain_classifier_layers=2,
        gradient_reversal_alpha=1.0,
        aggregate_output=False,
        proper_key_padding_mask=True,
        masked_domain_pooling=True,
    )
    adapter = CenteredHitAxisAdapter(base, 0.0)
    adapter.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return adapter.to(device).eval()


def centered_scores(hit_delta, global_logit_offset):
    return [
        (
            1 / (1 + np.exp(-np.clip(d - d.mean() + global_logit_offset, -40, 40)))
        ).astype(np.float32)
        for d in hit_delta
    ]


def set_summary(hit_delta):
    out = []
    for d in hit_delta:
        residual = d - d.mean()
        out.append(
            np.r_[
                np.quantile(residual, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]),
                residual.std(),
                len(d),
            ]
        )
    return np.asarray(out, np.float32)


def domain_probe(train_mc, train_exp, test_mc, test_exp, nonlinear=False):
    xtr = np.concatenate((train_mc, train_exp))
    ytr = np.r_[np.zeros(len(train_mc)), np.ones(len(train_exp))]
    xte = np.concatenate((test_mc, test_exp))
    yte = np.r_[np.zeros(len(test_mc)), np.ones(len(test_exp))]
    if nonlinear:
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=20,
            max_features="sqrt",
            random_state=42,
            n_jobs=4,
        )
    else:
        clf = make_pipeline(
            StandardScaler(), LogisticRegression(C=1, solver="liblinear")
        )
    clf.fit(xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(xte)[:, 1]))


def fixed_cohort_w1(
    mc_events, exp_events, original_mc, original_exp, centered_mc, centered_exp
):
    mc_keep = accepted(mc_events, original_mc) & accepted(mc_events, centered_mc)
    exp_keep = accepted(exp_events, original_exp) & accepted(exp_events, centered_exp)

    def w1(mc_scores, exp_scores):
        return float(
            wasserstein_distance(
                np.concatenate([s for s, keep in zip(mc_scores, mc_keep) if keep]),
                np.concatenate([s for s, keep in zip(exp_scores, exp_keep) if keep]),
            )
        )

    return {
        "n_mc": int(mc_keep.sum()),
        "n_exp": int(exp_keep.sum()),
        "w1_original": w1(original_mc, original_exp),
        "w1_centered": w1(centered_mc, centered_exp),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mc", default=MC_DEFAULT)
    parser.add_argument("--exp", default=EXP_DEFAULT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    threadpool_limits(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target = normalization(args.mc)
    events = {
        "mc_cal": load_events(args.mc, "val", 5000, target, True),
        "exp_cal": load_events(args.exp, "val", 5000, target, False),
        "mc_first": load_events(args.mc, "val", 5000, target, True, 5000),
        "exp_first": load_events(args.exp, "test", 5000, target, False),
        "mc_test": load_events(args.mc, "val", 5000, target, True, 10000),
        "exp_test": load_events(args.exp, "test", 5000, target, False, 5000),
    }
    model = model_from_checkpoint(args.base, True, True).to(device).eval()
    delta = {}
    for name, ev in events.items():
        mean, hits = infer_delta(model, ev, device)
        delta[name] = {"mean": mean, "hits": hits}
        print("INFERRED", name, len(ev), flush=True)

    labels = np.concatenate([e["y"] for e in events["mc_cal"]])
    centered_logits = np.concatenate([d - d.mean() for d in delta["mc_cal"]["hits"]])
    precision, recall, threshold = precision_recall_curve(labels, centered_logits)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1[:-1]))
    offset = -float(threshold[best])
    calibration = {
        "global_logit_offset": offset,
        "mc_cal_f1": float(f1[best]),
        "mc_cal_ap_centered": float(average_precision_score(labels, centered_logits)),
    }
    print("CALIBRATION", json.dumps(calibration), flush=True)

    original_mc = scores_with_jitter(delta["mc_test"]["hits"], np.zeros(5000), 0)
    original_exp = scores_with_jitter(delta["exp_test"]["hits"], np.zeros(5000), 0)
    centered_mc = centered_scores(delta["mc_test"]["hits"], offset)
    centered_exp = centered_scores(delta["exp_test"]["hits"], offset)
    adapter = CenteredHitAxisAdapter(model, offset).to(device).eval()
    with torch.no_grad():
        batch_x, batch_mask, _ = make_batch(events["mc_test"][:32], device)
        adapted_logits, adapted_pooled = adapter(batch_x, batch_mask)
        adapted_score = torch.softmax(adapted_logits, dim=-1)[:, :, 1].cpu().numpy()
    adapter_score_max_abs_difference = max(
        float(np.max(np.abs(adapted_score[i, : len(e["x"])] - centered_mc[i])))
        for i, e in enumerate(events["mc_test"][:32])
    )
    pooled_spread = float((adapted_pooled - adapted_pooled[:1]).abs().max().cpu())
    original_mc_first = scores_with_jitter(delta["mc_first"]["hits"], np.zeros(5000), 0)
    original_exp_first = scores_with_jitter(
        delta["exp_first"]["hits"], np.zeros(5000), 0
    )
    centered_mc_first = centered_scores(delta["mc_first"]["hits"], offset)
    centered_exp_first = centered_scores(delta["exp_first"]["hits"], offset)
    summary_mc_cal = set_summary(delta["mc_cal"]["hits"])
    summary_exp_cal = set_summary(delta["exp_cal"]["hits"])
    summary_mc_test = set_summary(delta["mc_test"]["hits"])
    summary_exp_test = set_summary(delta["exp_test"]["hits"])
    result = {
        "protocol": {
            "calibration_mc_val_offset": 0,
            "calibration_exp_val_offset": 0,
            "holdout_mc_val_offset": 10000,
            "holdout_exp_test_offset": 5000,
            "events_per_domain_per_split": 5000,
            "representation": "single hit-task axis, event mean subtracted",
        },
        "calibration": calibration,
        "adapter_score_max_abs_difference": adapter_score_max_abs_difference,
        "adapter_pooled_max_abs_spread": pooled_spread,
        "original_hit_metrics": hit_metrics(
            events["mc_test"], events["exp_test"], original_mc, original_exp
        ),
        "centered_hit_metrics": hit_metrics(
            events["mc_test"], events["exp_test"], centered_mc, centered_exp
        ),
        "mean_pooled_linear_probe_auc": domain_probe(
            np.full((5000, 1), offset),
            np.full((5000, 1), offset),
            np.full((5000, 1), offset),
            np.full((5000, 1), offset),
        ),
        "mean_pooled_nonlinear_probe_auc": domain_probe(
            np.full((5000, 1), offset),
            np.full((5000, 1), offset),
            np.full((5000, 1), offset),
            np.full((5000, 1), offset),
            nonlinear=True,
        ),
        "first_half_original_hit_metrics": hit_metrics(
            events["mc_first"],
            events["exp_first"],
            original_mc_first,
            original_exp_first,
        ),
        "first_half_centered_hit_metrics": hit_metrics(
            events["mc_first"],
            events["exp_first"],
            centered_mc_first,
            centered_exp_first,
        ),
        "first_half_fixed_cohort_w1": fixed_cohort_w1(
            events["mc_first"],
            events["exp_first"],
            original_mc_first,
            original_exp_first,
            centered_mc_first,
            centered_exp_first,
        ),
        "second_half_fixed_cohort_w1": fixed_cohort_w1(
            events["mc_test"],
            events["exp_test"],
            original_mc,
            original_exp,
            centered_mc,
            centered_exp,
        ),
        "set_summary_linear_probe_auc": domain_probe(
            summary_mc_cal, summary_exp_cal, summary_mc_test, summary_exp_test
        ),
        "set_summary_forest_probe_auc": domain_probe(
            summary_mc_cal,
            summary_exp_cal,
            summary_mc_test,
            summary_exp_test,
            nonlinear=True,
        ),
        "residual_quantiles_linear_probe_auc": domain_probe(
            summary_mc_cal[:, :-1],
            summary_exp_cal[:, :-1],
            summary_mc_test[:, :-1],
            summary_exp_test[:, :-1],
        ),
        "residual_quantiles_forest_probe_auc": domain_probe(
            summary_mc_cal[:, :-1],
            summary_exp_cal[:, :-1],
            summary_mc_test[:, :-1],
            summary_exp_test[:, :-1],
            nonlinear=True,
        ),
    }
    print("RESULT", json.dumps(result), flush=True)
    args.output.write_text(json.dumps(result, indent=2))
    torch.save(adapter.state_dict(), args.output.with_suffix(".ckpt"))


if __name__ == "__main__":
    main()
