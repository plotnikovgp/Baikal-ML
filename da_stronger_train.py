"""Warm-start DA with a trained discriminator and independent feature probes.

Designed for the report-era 2020 MC/experimental noise-signal datasets. The
domain adversary sees only MC muatm versus experimental events, while the hit
classifier continues to train on all MC particle types. Validation is separate
from training and uses two disjoint portions of MC/experimental val splits.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pytorch_training"))
# The cluster62 Torch environment lacks optional graph-network dependencies
# imported by models/__init__.py; this experiment only needs encoder/layers.
models_package = types.ModuleType("models")
models_package.__path__ = [str(ROOT / "pytorch_training" / "models")]
sys.modules["models"] = models_package
from models.encoder import EncoderDomainAdaptation  # noqa: E402

MC_DEFAULT = "/home3/ivkhar/Baikal/data/normed/baikal_2020_sig-noise_mid-eq_normed.h5"
EXP_DEFAULT = "/home/plotnikovgp/tmp/exp_da_subset_20261015.h5"
BASE_DEFAULT = "/home/plotnikovgp/tmp/base_da_20261015.ckpt"


def normalization(path: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path) as f:
        return (
            f["norm_param/mean"][:].astype(np.float32),
            f["norm_param/std"][:].astype(np.float32),
        )


def load_events(
    path: str,
    split: str,
    n: int,
    target_norm: tuple[np.ndarray, np.ndarray],
    mc: bool,
    offset: int = 0,
) -> list[dict]:
    src_mean, src_std = normalization(path)
    dst_mean, dst_std = target_norm
    with h5py.File(path) as f:
        group = f[split]
        starts = group["ev_starts/data"][offset : offset + n + 1]
        n = len(starts) - 1
        lo, hi = int(starts[0]), int(starts[-1])
        values = group["data/data"][lo:hi].astype(np.float32)
        values = (values * src_std + src_mean - dst_mean) / dst_std
        channels = group["channels/data"][lo:hi]
        ids = group["ev_ids/data"][offset : offset + n]
        labels = group["labels/data"][lo:hi] if mc else None
        events = []
        for i in range(n):
            a, b = int(starts[i] - lo), int(starts[i + 1] - lo)
            if b - a < 8 or len(np.unique(channels[a:b] // 36)) < 2:
                continue
            e = {"x": values[a:b][:256], "channels": channels[a:b][:256], "id": ids[i]}
            if mc:
                e["y"] = (labels[a:b][:256] != 0).astype(np.int64)
            events.append(e)
    return events


def make_batch(events: list[dict], device: torch.device):
    length = max(len(e["x"]) for e in events)
    x = np.zeros((len(events), length, 5), np.float32)
    mask = np.zeros((len(events), length), bool)
    y = np.full((len(events), length), -100, np.int64)
    for i, event in enumerate(events):
        size = len(event["x"])
        x[i, :size] = event["x"]
        mask[i, :size] = True
        if "y" in event:
            y[i, :size] = event["y"]
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(mask).to(device),
        torch.from_numpy(y).to(device),
    )


def draw(pool: list[dict], n: int) -> list[dict]:
    return random.choices(pool, k=n)


def pool_hidden(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weight).sum(1) / weight.sum(1).clamp_min(1)


def score_wasserstein_1d(
    scores: torch.Tensor,
    mask: torch.Tensor,
    n_mc: int,
    n_quantiles: int = 128,
) -> torch.Tensor:
    """Differentiable quantile approximation to hit-score Wasserstein-1."""
    mc = torch.sort(scores[:n_mc][mask[:n_mc]])[0]
    exp = torch.sort(scores[n_mc:][mask[n_mc:]])[0]
    quantiles = torch.linspace(0, 1, n_quantiles, device=scores.device)
    mc_index = (quantiles * (len(mc) - 1)).long()
    exp_index = (quantiles * (len(exp) - 1)).long()
    return torch.mean(torch.abs(mc[mc_index] - exp[exp_index]))


@torch.no_grad()
def infer(model, events: list[dict], device, batch_size=32) -> dict:
    model.eval()
    features, heads, scores = [], [], []
    for start in range(0, len(events), batch_size):
        group = events[start : start + batch_size]
        x, mask, _ = make_batch(group, device)
        logits, domain_logits, hidden = model(x, mask, return_hidden_states=True)
        features.append(pool_hidden(hidden, mask).float().cpu().numpy())
        heads.append(torch.softmax(domain_logits, -1)[:, 1].float().cpu().numpy())
        score = torch.softmax(logits, -1)[:, :, 1].float().cpu().numpy()
        scores.extend(score[i, : len(e["x"])] for i, e in enumerate(group))
    return {
        "features": np.concatenate(features),
        "heads": np.concatenate(heads),
        "scores": scores,
    }


def accepted(events: list[dict], scores: list[np.ndarray]) -> np.ndarray:
    out = []
    for event, score in zip(events, scores):
        keep = score >= 0.5
        out.append(
            keep.sum() >= 8 and len(np.unique(event["channels"][keep] // 36)) >= 2
        )
    return np.asarray(out)


def validate(model, validation: dict, device) -> dict:
    pred = {name: infer(model, events, device) for name, events in validation.items()}
    train_mc, train_exp = pred["mc_probe"], pred["exp_probe"]
    test_mc, test_exp = pred["mc_test"], pred["exp_test"]
    xtr = np.concatenate([train_mc["features"], train_exp["features"]])
    ytr = np.r_[
        np.zeros(len(train_mc["features"])), np.ones(len(train_exp["features"]))
    ]
    xte = np.concatenate([test_mc["features"], test_exp["features"]])
    yte = np.r_[np.zeros(len(test_mc["features"])), np.ones(len(test_exp["features"]))]
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    probe.fit(xtr, ytr)
    truth = np.concatenate([e["y"] for e in validation["mc_test"]])
    score = np.concatenate(test_mc["scores"])
    mc_cut = accepted(validation["mc_test"], test_mc["scores"])
    exp_cut = accepted(validation["exp_test"], test_exp["scores"])
    result = {
        "probe_auc": float(roc_auc_score(yte, probe.predict_proba(xte)[:, 1])),
        "domain_head_auc": float(
            roc_auc_score(yte, np.r_[test_mc["heads"], test_exp["heads"]])
        ),
        "mc_hit_auc": float(roc_auc_score(truth, score)),
        "mc_hit_ap": float(average_precision_score(truth, score)),
        "mc_hit_precision_p0p5": float(
            np.sum((score >= 0.5) & (truth == 1)) / max(1, np.sum(score >= 0.5))
        ),
        "mc_hit_recall_p0p5": float(
            np.sum((score >= 0.5) & (truth == 1)) / max(1, np.sum(truth == 1))
        ),
        "w1_all_hits": float(
            wasserstein_distance(score, np.concatenate(test_exp["scores"]))
        ),
        "n_mc_pass": int(mc_cut.sum()),
        "n_exp_pass": int(exp_cut.sum()),
    }
    if mc_cut.any() and exp_cut.any():
        mc_selected = np.concatenate(
            [s for s, k in zip(test_mc["scores"], mc_cut) if k]
        )
        exp_selected = np.concatenate(
            [s for s, k in zip(test_exp["scores"], exp_cut) if k]
        )
        result["w1_predicted_2s8h"] = float(
            wasserstein_distance(mc_selected, exp_selected)
        )
    return result


def model_from_checkpoint(
    path: str,
    proper_mask: bool,
    masked_pool: bool,
    signal_weighted_pool: bool = False,
):
    model = EncoderDomainAdaptation(
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
        proper_key_padding_mask=proper_mask,
        masked_domain_pooling=masked_pool,
        signal_weighted_domain_pooling=signal_weighted_pool,
    )
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=False), strict=True
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc", default=MC_DEFAULT)
    parser.add_argument("--exp", default=EXP_DEFAULT)
    parser.add_argument("--base", default=BASE_DEFAULT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--domain-batch", type=int, default=12)
    parser.add_argument("--lambda-da", type=float, default=0.02)
    parser.add_argument("--output-swd-k", type=float, default=0.0)
    parser.add_argument("--feature-mean-k", type=float, default=0.0)
    parser.add_argument("--lr-encoder", type=float, default=2e-5)
    parser.add_argument("--lr-domain", type=float, default=1e-3)
    parser.add_argument("--discriminator-steps", type=int, default=2)
    parser.add_argument("--fresh-discriminator-batches", action="store_true")
    parser.add_argument("--proper-mask", action="store_true")
    parser.add_argument("--masked-pool", action="store_true")
    parser.add_argument("--signal-weighted-pool", action="store_true")
    parser.add_argument("--domain-predicted-cut", action="store_true")
    parser.add_argument("--head-only", action="store_true")
    parser.add_argument("--mc-train-events", type=int, default=120000)
    parser.add_argument("--exp-train-events", type=int, default=60000)
    parser.add_argument("--probe-events", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device(args.device)
    target_norm = normalization(args.mc)
    print("loading training data", flush=True)
    mc_train = load_events(args.mc, "train", args.mc_train_events, target_norm, True)
    mc_domain = [e for e in mc_train if e["id"].startswith(b"muatm_")]
    exp_train = load_events(
        args.exp, "train", args.exp_train_events, target_norm, False
    )
    size = args.probe_events
    validation = {
        "mc_probe": load_events(args.mc, "val", size, target_norm, True),
        "mc_test": load_events(args.mc, "val", size, target_norm, True, size),
        "exp_probe": load_events(args.exp, "val", size, target_norm, False),
        "exp_test": load_events(args.exp, "val", size, target_norm, False, size),
    }
    print(
        "counts",
        {
            "mc_train": len(mc_train),
            "mc_domain": len(mc_domain),
            "exp_train": len(exp_train),
            **{k: len(v) for k, v in validation.items()},
        },
        flush=True,
    )
    model = model_from_checkpoint(
        args.base, args.proper_mask, args.masked_pool, args.signal_weighted_pool
    ).to(device)
    if args.domain_predicted_cut:
        print("filtering domain pools by base predicted 2s8h", flush=True)
        mc_keep = accepted(mc_domain, infer(model, mc_domain, device)["scores"])
        exp_keep = accepted(exp_train, infer(model, exp_train, device)["scores"])
        mc_domain = [e for e, k in zip(mc_domain, mc_keep) if k]
        exp_train = [e for e, k in zip(exp_train, exp_keep) if k]
        print("domain_pool_counts", len(mc_domain), len(exp_train), flush=True)
        if min(len(mc_domain), len(exp_train)) < args.domain_batch:
            raise ValueError("too few predicted-cut domain events")
    if args.head_only:
        for param in model.encoder.parameters():
            param.requires_grad_(False)
    encoder_params = list(model.encoder.parameters()) + list(
        model.main_head.parameters()
    )
    opt_enc = torch.optim.AdamW(encoder_params, lr=args.lr_encoder)
    opt_dom = torch.optim.AdamW(model.domain_classifier.parameters(), lr=args.lr_domain)

    def report(step: int) -> None:
        metrics = validate(model, validation, device)
        record = {"step": step, **metrics}
        with (args.out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        torch.save(model.state_dict(), args.out / f"step_{step:04d}.ckpt")
        print("EVAL", json.dumps(record), flush=True)

    report(0)
    for step in range(1, args.steps + 1):
        model.train()
        d_loss = torch.tensor(0.0, device=device)
        if (
            args.discriminator_steps
            or args.lambda_da
            or args.output_swd_k
            or args.feature_mean_k
        ):
            sampled_mc = draw(mc_domain, args.domain_batch)
            sampled_exp = draw(exp_train, args.domain_batch)
            dx, dm, _ = make_batch(sampled_mc + sampled_exp, device)
            dy = torch.cat(
                [
                    torch.zeros(args.domain_batch, dtype=torch.long, device=device),
                    torch.ones(args.domain_batch, dtype=torch.long, device=device),
                ]
            )
            if args.discriminator_steps and not args.fresh_discriminator_batches:
                with torch.no_grad():
                    _, detached_hidden = model.encoder(dx, dm)
                    if model.signal_weighted_domain_pooling:
                        hit_scores = torch.softmax(
                            model.main_head(detached_hidden), -1
                        )[:, :, 1]
                        weights = dm.to(detached_hidden.dtype) * hit_scores
                        weights = weights.unsqueeze(-1)
                        detached_features = (
                            (detached_hidden * weights).sum(1)
                            / weights.sum(1).clamp_min(1e-6)
                        ).detach()
                    else:
                        detached_features = pool_hidden(detached_hidden, dm).detach()
                for _ in range(args.discriminator_steps):
                    opt_dom.zero_grad(set_to_none=True)
                    d_loss = F.cross_entropy(
                        model.domain_classifier(detached_features), dy
                    )
                    d_loss.backward()
                    opt_dom.step()
            elif args.discriminator_steps:
                for _ in range(args.discriminator_steps):
                    fresh_mc = draw(mc_domain, args.domain_batch)
                    fresh_exp = draw(exp_train, args.domain_batch)
                    fx, fm, _ = make_batch(fresh_mc + fresh_exp, device)
                    with torch.no_grad():
                        _, detached_hidden = model.encoder(fx, fm)
                        detached_features = pool_hidden(detached_hidden, fm).detach()
                    opt_dom.zero_grad(set_to_none=True)
                    d_loss = F.cross_entropy(
                        model.domain_classifier(detached_features), dy
                    )
                    d_loss.backward()
                    opt_dom.step()

        for param in model.domain_classifier.parameters():
            param.requires_grad_(False)
        sx, sm, sy = make_batch(draw(mc_train, args.batch), device)
        opt_enc.zero_grad(set_to_none=True)
        _, cls_hidden = model.encoder(sx, sm)
        cls_logits = model.main_head(cls_hidden)
        cls_loss = F.cross_entropy(cls_logits[sm], sy[sm])
        adv_loss = torch.tensor(0.0, device=device)
        if args.lambda_da:
            _, domain_logits = model(dx, dm)
            adv_loss = F.cross_entropy(domain_logits, dy)
        swd_loss = torch.tensor(0.0, device=device)
        mean_loss = torch.tensor(0.0, device=device)
        if args.output_swd_k or args.feature_mean_k:
            _, domain_hidden = model.encoder(dx, dm)
            if args.output_swd_k:
                domain_scores = torch.softmax(model.main_head(domain_hidden), -1)[
                    :, :, 1
                ]
                swd_loss = score_wasserstein_1d(domain_scores, dm, args.domain_batch)
            if args.feature_mean_k:
                pooled = pool_hidden(domain_hidden, dm)
                mean_loss = torch.mean(
                    (
                        pooled[: args.domain_batch].mean(0)
                        - pooled[args.domain_batch :].mean(0)
                    )
                    ** 2
                )
        (
            cls_loss
            + args.lambda_da * adv_loss
            + args.output_swd_k * swd_loss
            + args.feature_mean_k * mean_loss
        ).backward()
        torch.nn.utils.clip_grad_norm_(encoder_params, 5.0)
        opt_enc.step()
        for param in model.domain_classifier.parameters():
            param.requires_grad_(True)
        if step % 20 == 0:
            print(
                "TRAIN",
                step,
                float(cls_loss.detach()),
                float(d_loss.detach()),
                float(adv_loss.detach()),
                float(swd_loss.detach()),
                float(mean_loss.detach()),
                flush=True,
            )
        if step % args.eval_every == 0 or step == args.steps:
            report(step)


if __name__ == "__main__":
    main()
