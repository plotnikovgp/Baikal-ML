"""Evaluate DA candidates on untouched MC val and experimental test events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from da_stronger_train import (
    EXP_DEFAULT,
    MC_DEFAULT,
    load_events,
    model_from_checkpoint,
    normalization,
    validate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mc", default=MC_DEFAULT)
    parser.add_argument("--exp", default=EXP_DEFAULT)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--mc-test-offset", type=int, default=10000)
    parser.add_argument("--exp-test-offset", type=int, default=5000)
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=4,
        metavar=("NAME", "PATH", "PROPER_MASK", "MASKED_POOL"),
        required=True,
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    target = normalization(args.mc)
    validation = {
        "mc_probe": load_events(args.mc, "val", args.n, target, True),
        "exp_probe": load_events(args.exp, "val", args.n, target, False),
        "mc_test": load_events(
            args.mc, "val", args.n, target, True, args.mc_test_offset
        ),
        "exp_test": load_events(
            args.exp, "test", args.n, target, False, args.exp_test_offset
        ),
    }
    print("counts", {k: len(v) for k, v in validation.items()}, flush=True)
    assert not {e["id"] for e in validation["mc_probe"]} & {
        e["id"] for e in validation["mc_test"]
    }
    assert not {e["id"] for e in validation["exp_probe"]} & {
        e["id"] for e in validation["exp_test"]
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result = {
        "protocol": {
            "n": args.n,
            "mc_test_offset": args.mc_test_offset,
            "exp_test_offset": args.exp_test_offset,
        },
        "models": {},
    }
    for name, path, proper, masked in args.candidate:
        model = model_from_checkpoint(
            path, proper.lower() == "true", masked.lower() == "true"
        ).to(device)
        metrics = validate(model, validation, device)
        result["models"][name] = metrics
        print("MODEL", name, json.dumps(metrics), flush=True)
        del model
        torch.cuda.empty_cache()
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
