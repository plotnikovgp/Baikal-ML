"""Domain separability in the one encoder direction needed by the hit head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from da_force_invariance import hit_metrics, hit_scores, linear_probe, pooled
from da_stronger_train import (
    EXP_DEFAULT,
    MC_DEFAULT,
    load_events,
    make_batch,
    model_from_checkpoint,
    normalization,
)


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
        "mc_probe_val": load_events(args.mc, "val", 5000, target, True),
        "exp_probe_val": load_events(args.exp, "val", 5000, target, False),
        "mc_probe_test": load_events(args.mc, "val", 5000, target, True, 5000),
        "exp_probe_test": load_events(args.exp, "test", 5000, target, False),
        "mc_holdout": load_events(args.mc, "val", 5000, target, True, 10000),
        "exp_holdout": load_events(args.exp, "test", 5000, target, False, 5000),
    }
    model = model_from_checkpoint(args.base, True, True).to(device).eval()
    weight = model.main_head.weight.detach().cpu().numpy()
    u = (weight[1] - weight[0]).astype(np.float64)
    u /= np.linalg.norm(u)
    features = {}
    for name, ev in events.items():
        features[name] = pooled(model, ev, device)
        print("FEATURES", name, features[name].shape, flush=True)

    def auc(names, hit_axis):
        train_mc, train_exp, test_mc, test_exp = (features[k] for k in names)
        if hit_axis:
            train_mc, train_exp, test_mc, test_exp = (
                (x @ u).reshape(-1, 1) for x in (train_mc, train_exp, test_mc, test_exp)
            )
        return linear_probe(train_mc, train_exp, test_mc, test_exp)[0]

    names = ("mc_probe_val", "exp_probe_val", "mc_holdout", "exp_holdout")
    calibrated = ("mc_probe_test", "exp_probe_test", "mc_holdout", "exp_holdout")
    result = {
        "ordinary_full_probe_auc": auc(names, False),
        "ordinary_hit_axis_probe_auc": auc(names, True),
        "calibrated_full_probe_auc": auc(calibrated, False),
        "calibrated_hit_axis_probe_auc": auc(calibrated, True),
    }
    # hit_scores expects columns of an orthonormal removed basis, so use an
    # explicit 1D projection here to verify task-score preservation.
    projected_scores = {}
    original_scores = {}
    for name in ("mc_holdout", "exp_holdout"):
        ev = events[name]
        original_scores[name] = hit_scores(
            model, model.main_head, ev, np.empty((128, 0)), device
        )
        scores = []
        with torch.no_grad():
            direction = torch.as_tensor(u, dtype=torch.float32, device=device)
            for start in range(0, len(ev), 32):
                group = ev[start : start + 32]
                x, mask, _ = make_batch(group, device)
                _, hidden = model.encoder(x, mask)
                hidden = (hidden @ direction).unsqueeze(-1) * direction
                score = (
                    torch.softmax(model.main_head(hidden), -1)[:, :, 1].cpu().numpy()
                )
                scores.extend(score[i, : len(e["x"])] for i, e in enumerate(group))
        projected_scores[name] = scores
    result["max_hit_score_abs_change"] = max(
        float(np.max(np.abs(x - y)))
        for name in ("mc_holdout", "exp_holdout")
        for x, y in zip(original_scores[name], projected_scores[name])
    )
    result["hit_metrics_original"] = hit_metrics(
        events["mc_holdout"],
        events["exp_holdout"],
        original_scores["mc_holdout"],
        original_scores["exp_holdout"],
    )
    result["hit_metrics_hit_axis_only"] = hit_metrics(
        events["mc_holdout"],
        events["exp_holdout"],
        projected_scores["mc_holdout"],
        projected_scores["exp_holdout"],
    )
    print("RESULT", json.dumps(result), flush=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
