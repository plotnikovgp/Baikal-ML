#!/usr/bin/env bash
set -euo pipefail

cd /home/plotnikovgp/tmp/angle_reconstruction/track_anchor

CUDA_VISIBLE_DEVICES=1 /home/plotnikovgp/tmp/ENTER/bin/python3.11 \
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
    --output-dir runs/mixed_universal/continue_v4_physics_graph_2k \
    --batch-size 128 --max-seq-len 128 --workers 2 \
    --steps 2000 --val-every 250 \
    --gt-weight 0.4 --point-weight 0.5 \
    --backbone-lr 5e-6 --point-lr 1e-5 \
    --max-pred-q68 4.9 --max-gt-q68 4.4 \
    --pred-score-weight 0.5 --muatm-score-weight 0.03 \
    --single-muatm-only \
    --pred-thinning-prob 0.5 --pred-threshold-min 0.7 --pred-threshold-max 0.9 \
    --graph-layers 2 --graph-k 12 \
    --graph-photon-speed-m-per-ns 0.225 --graph-minkowski-edge \
    --graph-only --graph-lr 1e-4 --init-from-transformer-model \
    --seed 137
