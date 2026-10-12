#!/usr/bin/env bash
set -euo pipefail

weight=${1:?Usage: run_point_balance_v4_3k.sh 0.2|0.8 GPU}
gpu=${2:-0}
case "$weight" in 0.2|0.8) ;; *) echo "Unsupported point weight: $weight" >&2; exit 2 ;; esac
cd /home/plotnikovgp/tmp/angle_reconstruction/track_anchor

CUDA_VISIBLE_DEVICES="$gpu" /home/plotnikovgp/tmp/ENTER/bin/python3.11 \
    train_mixed_universal.py \
    --pred-data /home/plotnikovgp/tmp/angle_reconstruction/data/baikal_2020_all_predsignal_p070_2s8h.h5 \
    --norm-data /home/plotnikovgp/tmp/angle_reconstruction/data/baikal_mc_merged2020_signal_8-2_angl_norm.h5 \
    --pred-anchors all_predsignal_p070_track_anchors_v1.h5 \
    --gt-data /home/plotnikovgp/tmp/angle_reconstruction/data/baikal_mc_merged2020_signal_8-2_angl_norm.h5 \
    --gt-anchors nue2_track_anchors_v3.h5 \
    --masks runs/mixed_universal/masks_v2.npz \
    --angle-checkpoint nue2_best_q68_trim_cap8.pt \
    --track-checkpoint runs/direct_line_from_best/best_line_q68.pt \
    --load-model runs/mixed_universal/continue_5k_stochastic_p/best.pt \
    --output-dir "runs/mixed_universal/continue_v4_point_weight_${weight}_3k" \
    --batch-size 256 --max-seq-len 128 --workers 2 \
    --steps 3000 --val-every 500 \
    --gt-weight 0.4 --point-weight "$weight" \
    --backbone-lr 5e-6 --point-lr 1e-5 \
    --max-pred-q68 4.9 --max-gt-q68 4.4 \
    --pred-score-weight 0.5 --muatm-score-weight 0.03 \
    --single-muatm-only \
    --pred-thinning-prob 0.5 --pred-threshold-min 0.7 --pred-threshold-max 0.9 \
    --seed 257
