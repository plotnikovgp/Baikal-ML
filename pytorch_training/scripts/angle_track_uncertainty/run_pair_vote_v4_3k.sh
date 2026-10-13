#!/usr/bin/env bash
set -euo pipefail

gpu=${1:-0}
cd /home/plotnikovgp/tmp/angle_reconstruction/track_anchor
CUDA_VISIBLE_DEVICES="$gpu" /home/plotnikovgp/tmp/ENTER/bin/python3.11 \
    train_pair_vote_residual.py \
    --bundle runs/mixed_universal/universal_angle_track_uncertainty_v4.pt \
    --pred-data /home/plotnikovgp/tmp/angle_reconstruction/data/baikal_2020_all_predsignal_p070_2s8h.h5 \
    --pred-anchors all_predsignal_p070_track_anchors_v1.h5 \
    --gt-data /home/plotnikovgp/tmp/angle_reconstruction/data/baikal_mc_merged2020_signal_8-2_angl_norm.h5 \
    --gt-anchors nue2_track_anchors_v3.h5 \
    --masks runs/mixed_universal/masks_v2.npz \
    --output-dir runs/pair_vote_v4/frozen_base_3k_gt24 \
    --steps 3000 --val-every 500 \
    --batch-size 128 --max-seq-len 128 --workers 2 \
    --sample-hits 32 --learning-rate 0.001 --gt-weight 0.4 \
    --gt-val-stride 24 --seed 573
