import argparse
import sys
import time
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import load_model
from train import create_dataloaders_from_config, create_optimizer, fix_seed, setup_perf
from train_types import get_train_type


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=30)
    parser.add_argument("--trace", default="profile_noise_sig_trace.json")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def sync_if_cuda(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def run_step(model, train_type, optimizer, train_iter, device: str):
    timings = {}

    sync_if_cuda(device)
    t0 = time.perf_counter()
    with record_function("dataloader_next"):
        data = next(train_iter)
    sync_if_cuda(device)
    timings["data_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with record_function("forward_loss"):
        result = train_type.process_batch(model, data)
        loss = train_type.get_criterion()(result["output"], result["y_true"])
        if isinstance(loss, dict):
            loss = loss["loss"]
    sync_if_cuda(device)
    timings["forward_loss_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with record_function("backward_step"):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    sync_if_cuda(device)
    timings["backward_step_s"] = time.perf_counter() - t0
    timings["loss"] = float(loss.detach().cpu())
    return timings


def main():
    args = parse_args()
    config_dir = str(Path(__file__).resolve().parents[1] / "conf")
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hydra.compose(config_name="config", overrides=args.overrides)
    OmegaConf.set_struct(cfg, False)

    fix_seed(cfg.random_seed, deterministic=bool(cfg.get("perf", {}).get("deterministic", False)))
    setup_perf(cfg)

    device = args.device
    train_params = OmegaConf.to_container(cfg, resolve=True)
    train_params["train_type_name"] = cfg.train_type.name
    train_params["is_graph"] = cfg.train_type.is_graph
    train_params["val_mode"] = False
    train_type = get_train_type(cfg.train_type.name, train_params, device)

    dataloaders, _ = create_dataloaders_from_config(cfg, train_type)
    model = load_model(cfg.model.type, OmegaConf.to_container(cfg.model.params, resolve=True)).to(
        device
    )
    model.train()
    optimizer = create_optimizer(model, cfg)
    train_iter = dataloaders["train"]

    for _ in range(args.warmup_steps):
        run_step(model, train_type, optimizer, train_iter, device)

    activities = [ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)

    totals = {"data_s": 0.0, "forward_loss_s": 0.0, "backward_step_s": 0.0}
    with profile(activities=activities, record_shapes=False) as prof:
        for _ in range(args.profile_steps):
            step_timings = run_step(model, train_type, optimizer, train_iter, device)
            for key in totals:
                totals[key] += step_timings[key]
            prof.step()

    print("Average step timings:")
    for key, value in totals.items():
        print(f"  {key}: {value / args.profile_steps:.4f}s")
    print(f"  total: {sum(totals.values()) / args.profile_steps:.4f}s")

    sort_key = "self_cuda_time_total" if device.startswith("cuda") else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=25))
    prof.export_chrome_trace(args.trace)
    print(f"Trace written to {args.trace}")


if __name__ == "__main__":
    main()
