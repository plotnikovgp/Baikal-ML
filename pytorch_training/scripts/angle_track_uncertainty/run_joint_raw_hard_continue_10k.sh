#!/usr/bin/env bash
set -euo pipefail

gpu=${1:-0}
cd /home/plotnikovgp/tmp/angle_reconstruction/track_anchor
CUDA_VISIBLE_DEVICES="$gpu" /home/plotnikovgp/tmp/ENTER/bin/python3.11 \
    train_joint_raw.py \
    --raw-mc /home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_2020_sig-noise_mid-eq_normed.h5 \
    --index joint_raw_index_v1.npz \
    --signal-checkpoint /home/plotnikovgp/baikal/Baikal-ML/pytorch_training/checkpoints/noise_sig_experiments/noise_sig_tres_abs_merged/best.ckpt \
    --model-norm-source /home/plotnikovgp/baikal/data/baikal_mc_merged_m50-na15-nue15_all_smaller_norm.h5 \
    --output-dir runs/joint_raw/pretrained_hard_continue_10k \
    --resume runs/joint_raw/pretrained_hard_5k/best.pt \
    --hard-padding-mask \
    --batch-events 128 --max-seq-len 256 --workers 2 \
    --steps 10000 --warm-steps 1000 --val-every 500 \
    --backbone-lr 1e-5 --head-lr 2e-4 \
    --point-weight 0.5 --hit-weight 0.1 --muatm-angle-weight 4.0 \
    --seed 422
