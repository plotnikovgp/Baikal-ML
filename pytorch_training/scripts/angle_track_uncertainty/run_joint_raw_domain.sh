#!/usr/bin/env bash
set -euo pipefail

checkpoint=${1:?Usage: run_joint_raw_domain.sh CHECKPOINT TAG GPU}
tag=${2:?Usage: run_joint_raw_domain.sh CHECKPOINT TAG GPU}
gpu=${3:-0}
cd /home/plotnikovgp/tmp/angle_reconstruction/track_anchor
output="runs/joint_raw/${tag}_domain"
mkdir -p "$output"

for split in domain_mc exp; do
    if [[ "$split" == exp ]]; then
        source=/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_exp_filtered_norm_w_model_enc_v2_fix_q.h5
    else
        source=/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_2020_sig-noise_mid-eq_normed.h5
    fi
    CUDA_VISIBLE_DEVICES="$gpu" /home/plotnikovgp/tmp/ENTER/bin/python3.11 \
        infer_joint_raw_domain.py \
        --checkpoint "$checkpoint" \
        --raw-data "$source" \
        --index joint_raw_domain_v1.npz \
        --split "$split" \
        --model-norm-source /home/plotnikovgp/baikal/data/baikal_mc_merged_m50-na15-nue15_all_smaller_norm.h5 \
        --output "$output/${split}.csv" \
        --batch-events 128 --max-seq-len 256 --threads 4
done
