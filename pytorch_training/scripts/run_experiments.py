#!/usr/bin/env python
import argparse
import copy
import multiprocessing
import os
import subprocess
import time
from pathlib import Path

import yaml


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(config, output_path):
    with open(output_path, "w") as f:
        yaml.dump(config, f)


def run_training(config_path, gpu_id=None, disable_wandb=False):
    env = os.environ.copy()

    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = ["python", "train.py", "-c", config_path]
    if disable_wandb:
        cmd.append("-dw")

    process = subprocess.Popen(cmd, env=env)
    return process


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple experiments with parameter variations"
    )
    parser.add_argument(
        "--base-config",
        "-c",
        default="train_configs/anle_multi_da.yaml",
        help="Base configuration file path",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="5,6",
        help="Comma-separated list of GPU IDs to use (default: use all available)",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=4,
        help="Maximum number of parallel experiments (default: number of GPUs)",
    )
    parser.add_argument(
        "--disable-wandb",
        "-dw",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--output-dir",
        default="temp_configs",
        help="Directory to save modified config files",
    )
    args = parser.parse_args()

    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpus.split(",")]

    max_parallel = args.max_parallel
    if max_parallel is None:
        if gpu_ids:
            max_parallel = len(gpu_ids)
        else:
            max_parallel = multiprocessing.cpu_count() - 2

    base_config_path = args.base_config

    parameter_variations = [
        # Vary hidden size and dff and num_layers
        {
            "model_params": {"gradient_reversal_alpha": 0.0},
            "exp_name": "encoder_nl5_hs512_dff512_nh1_lr1e4_bs256_nm250_wr5_wm5_da_k01_gr01_dp0_not_pretrained",
        },
        # {"model_params": {"hidden_size": 128, "dim_feedforward_size": 128, "num_layers": 5}, "exp_name": "encoder_hs128_dff128_nl5"},
        # {"model_params": {"hidden_size": 128, "dim_feedforward_size": 512, "num_layers": 5}, "exp_name": "encoder_hs128_dff512_nl5"},
        # {"model_params": {"hidden_size": 128, "dim_feedforward_size": 512, "num_layers": 3}, "exp_name": "encoder_hs128_dff512_nl3"},
        # {"model_params": {"hidden_size": 512, "dim_feedforward_size": 512, "num_layers": 5}, "exp_name": "encoder_hs512_dff512_nl5"},
        # {"model_params": {"hidden_size": 512, "dim_feedforward_size": 512, "num_layers": 7}, "exp_name": "encoder_hs512_dff512_nl7"},
    ]

    base_config = load_config(base_config_path)

    # Create a temp directory for configs if it doesn't exist
    temp_config_dir = Path(args.output_dir)
    temp_config_dir.mkdir(exist_ok=True)

    running_processes = {}
    total_experiments = len(parameter_variations)
    experiments_completed = 0

    # Print experiment plan
    print(f"Running {total_experiments} experiments with maximum {max_parallel} in parallel")
    if gpu_ids:
        print(f"Using GPUs: {gpu_ids}")
    else:
        print("Using all available GPUs")

    # Process experiments
    exp_idx = 0
    while experiments_completed < total_experiments:
        # Start new experiments if we have capacity
        while exp_idx < total_experiments and len(running_processes) < max_parallel:
            variation = parameter_variations[exp_idx]
            config = copy.deepcopy(base_config)

            exp_name = variation.pop("exp_name")
            config["exp_name"] = exp_name

            # Apply parameter updates
            for param, value in variation.items():
                if isinstance(value, dict) and param in config and isinstance(config[param], dict):
                    # For nested parameters like model_params
                    config[param].update(value)
                else:
                    # For top-level parameters
                    config[param] = value

            temp_config_path = temp_config_dir / f"{exp_name}.yaml"
            save_config(config, temp_config_path)

            gpu_id = None
            if gpu_ids:
                gpu_id = gpu_ids[len(running_processes) % len(gpu_ids)]

            print(f"Starting experiment [{exp_idx + 1}/{total_experiments}]: {exp_name}")
            process = run_training(str(temp_config_path), gpu_id, args.disable_wandb)
            running_processes[exp_name] = process
            exp_idx += 1

        # Check for completed processes
        completed_experiments = []
        for exp_name, process in running_processes.items():
            if process.poll() is not None:
                exit_code = process.returncode
                status = (
                    "completed successfully" if exit_code == 0 else f"failed with code {exit_code}"
                )
                print(f"Experiment {exp_name} {status}")
                completed_experiments.append(exp_name)
                experiments_completed += 1

        # Remove completed processes from tracking
        for exp_name in completed_experiments:
            del running_processes[exp_name]

        # Sleep briefly to avoid CPU thrashing
        if running_processes and exp_idx < total_experiments:
            time.sleep(5)

    print(f"\nAll {total_experiments} experiments have been processed.")


if __name__ == "__main__":
    main()
