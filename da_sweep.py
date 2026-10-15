"""Small controlled DA continuation with per-epoch frozen logistic-regression probes.

Four variants share one initialization and exactly the same disjoint data split.
No checkpoint is selected using the independent probe test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from da_probe import CKPT_DIR, EXP, PARTICLES, SCORES, SRC, evaluate, load_events, norm
from models.encoder import EncoderDomainAdaptation

INITIAL = CKPT_DIR / "k_nsol_labelneq0_da_hs128_k0p0001"
VARIANTS = {
    "control": {"alpha": 0.0, "disc_lr": 1e-3, "masked": True},
    "da_unmasked": {"alpha": 0.1, "disc_lr": 1e-3, "masked": False},
    "da_masked": {"alpha": 0.1, "disc_lr": 1e-3, "masked": True},
    "da_masked_fastdisc": {"alpha": 0.3, "disc_lr": 3e-3, "masked": True},
    "da_masked_lr_only": {"alpha": 0.1, "disc_lr": 3e-3, "masked": True},
    "da_masked_alpha_only": {"alpha": 0.3, "disc_lr": 1e-3, "masked": True},
}


def build_data(n_probe: int, n_train_per_particle: int, paper_like: bool):
    rng = np.random.default_rng(41324)
    target_norm = norm(SRC)
    probe = {}
    train_mc = []
    for particle in PARTICLES:
        factor = (1.5 if particle == "muatm" else 0.75) if paper_like else 1.0
        particle_probe = round(n_probe * factor)
        particle_train = round(n_train_per_particle * factor)
        path = SCORES / f"2020_{particle}_scores_allhits_spaced20k.h5"
        order = rng.permutation(20_000)
        probe[f"{particle}_train"] = load_events(
            path, "test", np.sort(order[:particle_probe]), target_norm, "mc"
        )
        probe[f"{particle}_test"] = load_events(
            path,
            "test",
            np.sort(order[particle_probe : 2 * particle_probe]),
            target_norm,
            "mc",
        )
        train_mc.append(
            load_events(
                path,
                "test",
                np.sort(
                    order[2 * particle_probe : 2 * particle_probe + particle_train]
                ),
                target_norm,
                "mc",
            )
        )
        assert not set(train_mc[-1]["ids"]) & set(probe[f"{particle}_train"]["ids"])
        assert not set(train_mc[-1]["ids"]) & set(probe[f"{particle}_test"]["ids"])
    for split, key in (("val", "exp_train"), ("test", "exp_test")):
        with h5py.File(EXP) as f:
            amount = len(f[f"{split}/ev_starts/data"]) - 1
        n_exp_probe = sum(len(probe[f"{p}_train"]["x"]) for p in PARTICLES)
        probe[key] = load_events(
            EXP,
            split,
            np.sort(rng.choice(amount, n_exp_probe, replace=False)),
            target_norm,
            "exp",
        )
    with h5py.File(EXP) as f:
        amount = len(f["train/ev_starts/data"]) - 1
    n_mc_train = sum(len(v["x"]) for v in train_mc)
    train_exp = load_events(
        EXP,
        "train",
        np.sort(rng.choice(amount, n_mc_train, replace=False)),
        target_norm,
        "exp",
    )
    assert not set(train_exp["ids"]) & set(probe["exp_train"]["ids"])
    assert not set(train_exp["ids"]) & set(probe["exp_test"]["ids"])
    mc = {k: np.concatenate([v[k] for v in train_mc]) for k in ("x", "mask")}
    mc["particle"] = np.concatenate(
        [np.full(len(v["x"]), i, dtype=np.int8) for i, v in enumerate(train_mc)]
    )
    labels = np.zeros(mc["mask"].shape, dtype=np.int64)
    pos = 0
    for particle_data in train_mc:
        for label in particle_data["labels"]:
            labels[pos, : len(label)] = label
            pos += 1
    mc["labels"] = labels
    return mc, train_exp, probe


def train_epoch(
    model,
    optimizer,
    mc,
    exp,
    batch_size,
    masked,
    device,
    rng,
    z_flip=False,
    domain_muatm_only=False,
):
    model.train()
    mc_order = rng.permutation(len(mc["x"]))
    ex_order = rng.permutation(len(exp["x"]))
    half = batch_size // 2
    muatm_indices = np.flatnonzero(mc["particle"] == 0) if domain_muatm_only else None
    sums = np.zeros(3, dtype=float)
    steps = 0
    for start in range(0, min(len(mc_order), len(ex_order)), half):
        imc, iex = mc_order[start : start + half], ex_order[start : start + half]
        if len(imc) != half or len(iex) != half:
            break
        if domain_muatm_only:
            idomain = rng.choice(muatm_indices, half, replace=True)
            x_array = np.concatenate([mc["x"][imc], mc["x"][idomain], exp["x"][iex]])
            mask_array = np.concatenate(
                [mc["mask"][imc], mc["mask"][idomain], exp["mask"][iex]]
            )
        else:
            x_array = np.concatenate([mc["x"][imc], exp["x"][iex]])
            mask_array = np.concatenate([mc["mask"][imc], exp["mask"][iex]])
        x = torch.from_numpy(x_array).to(device)
        if z_flip:
            x = x.clone()
            sign = torch.where(torch.rand(len(x), device=device) < 0.5, -1.0, 1.0)
            x[:, :, 4] *= sign[:, None]
        mask = torch.from_numpy(mask_array).to(device)
        domain = torch.cat(
            (torch.zeros(half, dtype=torch.long), torch.ones(half, dtype=torch.long))
        ).to(device)
        y = torch.from_numpy(mc["labels"][imc]).to(device)
        if masked:
            _, h = model.encoder(x, mask)
            hit_logits = model.main_head(h)
            pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            domain_logits = model.domain_classifier(model.gradient_reversal(pooled))
        else:
            hit_logits, domain_logits = model(x, mask)
        main_loss = F.cross_entropy(hit_logits[:half][mask[:half]], y[mask[:half]])
        domain_loss = F.cross_entropy(
            domain_logits[half:] if domain_muatm_only else domain_logits, domain
        )
        loss = main_loss + domain_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        sums += [
            float(loss.detach()),
            float(main_loss.detach()),
            float(domain_loss.detach()),
        ]
        steps += 1
    return {
        k: float(v / steps)
        for k, v in zip(("total_loss", "hit_loss", "domain_loss"), sums)
    } | {"steps": steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--n-probe", type=int, default=600)
    parser.add_argument("--n-train-per-particle", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--paper-like",
        action="store_true",
        help="2:1:1 MC mix and random z flip during training",
    )
    parser.add_argument(
        "--domain-muatm-only",
        action="store_true",
        help="use separate muatm MC stream only for the domain loss",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    mc, exp, probe = build_data(
        args.n_probe, args.n_train_per_particle, args.paper_like
    )
    print(
        "loaded",
        len(mc["x"]),
        len(exp["x"]),
        {k: len(v["x"]) for k, v in probe.items()},
        flush=True,
    )
    with open(INITIAL / "train_config_mc_2020.yaml") as f:
        config = yaml.safe_load(f)
    model = EncoderDomainAdaptation(**config["model"]["params"])
    model.load_state_dict(
        torch.load(
            INITIAL / "best_mc_2020.ckpt", map_location="cpu", weights_only=False
        )
    )
    variant = VARIANTS[args.variant]
    model.gradient_reversal.alpha = variant["alpha"]
    model.to(device)
    main_params = list(model.encoder.parameters()) + list(model.main_head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": main_params, "lr": 1e-4},
            {"params": model.domain_classifier.parameters(), "lr": variant["disc_lr"]},
        ],
        weight_decay=0.01,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or args.variant
    report = {
        "variant": args.variant,
        "seed": args.seed,
        "paper_like_mix_and_z_flip": args.paper_like,
        "settings": variant,
        "domain_muatm_only": args.domain_muatm_only,
        "main_lr": 1e-4,
        "n_train_mc": len(mc["x"]),
        "n_train_exp": len(exp["x"]),
        "epochs": [],
    }
    for epoch in range(args.epochs + 1):
        if epoch:
            train_metrics = train_epoch(
                model,
                optimizer,
                mc,
                exp,
                args.batch_size,
                variant["masked"],
                device,
                rng,
                args.paper_like,
                args.domain_muatm_only,
            )
        else:
            train_metrics = None
        val = evaluate(
            model, probe, args.batch_size, device, masked_head=variant["masked"]
        )
        row = {"epoch": epoch, "train": train_metrics, "validation": val}
        report["epochs"].append(row)
        (args.output_dir / f"{run_name}.json").write_text(json.dumps(report, indent=2))
        print(
            "epoch",
            epoch,
            "train",
            train_metrics,
            "mixed",
            val["mixed"],
            "muatm",
            val["muatm"],
            "hit",
            val["hit_task"],
            flush=True,
        )
        if epoch:
            torch.save(
                model.state_dict(), args.output_dir / f"{run_name}_e{epoch}.ckpt"
            )


if __name__ == "__main__":
    main()
