#!/usr/bin/env bash
set -euo pipefail

variant=${1:?Usage: run_joint_raw_5k.sh legacy|hard GPU}
gpu=${2:-0}
case "$variant" in legacy|hard) ;; *) echo "Unknown variant: $variant" >&2; exit 2 ;; esac
cd /home/plotnikovgp/tmp/angle_reconstruction/track_anchor

extra=()
if [[ "$variant" == hard ]]; then
    extra+=(--hard-padding-mask)
fi

CUDA_VISIBLE_DEVICES="$gpu" /home/plotnikovgp/tmp/ENTER/bin/python3.11 \
    train_joint_raw.py \
    --raw-mc /home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_2020_sig-noise_mid-eq_normed.h5 \
    --index joint_raw_index_v1.npz \
    --signal-checkpoint /home/plotnikovgp/baikal/Baikal-ML/pytorch_training/checkpoints/noise_sig_experiments/noise_sig_tres_abs_merged/best.ckpt \
    --model-norm-source /home/plotnikovgp/baikal/data/baikal_mc_merged_m50-na15-nue15_all_smaller_norm.h5 \
    --output-dir "runs/joint_raw/pretrained_${variant}_5k" \
    --batch-events 128 --max-seq-len 256 --workers 2 \
    --steps 5000 --warm-steps 1000 --val-every 500 \
    --backbone-lr 1e-5 --head-lr 3e-4 \
    --point-weight 0.5 --hit-weight 0.1 --muatm-angle-weight 4.0 \
    --seed 421 "${extra[@]}"
