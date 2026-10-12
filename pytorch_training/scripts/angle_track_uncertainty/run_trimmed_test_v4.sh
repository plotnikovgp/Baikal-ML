#!/usr/bin/env bash
set -euo pipefail

cd /home/plotnikovgp/tmp/angle_reconstruction/track_anchor
python=/home/plotnikovgp/tmp/ENTER/bin/python3.11
pred_data=/home/plotnikovgp/tmp/angle_reconstruction/data/baikal_2020_all_predsignal_p070_2s8h.h5
gt_data=/home/plotnikovgp/tmp/angle_reconstruction/data/baikal_mc_merged2020_signal_8-2_angl_norm.h5
root=runs/mixed_universal/continue_v4_trimmed_pred_2k
out=$root/test_best

CUDA_VISIBLE_DEVICES=1 "$python" train_mixed_universal.py \
    --pred-data "$pred_data" --norm-data "$gt_data" \
    --pred-anchors all_predsignal_p070_track_anchors_v1.h5 \
    --gt-data "$gt_data" --gt-anchors nue2_track_anchors_v3.h5 \
    --masks runs/mixed_universal/masks_v2.npz \
    --angle-checkpoint nue2_best_q68_trim_cap8.pt \
    --track-checkpoint runs/direct_line_from_best/best_line_q68.pt \
    --load-model "$root/best.pt" --output-dir "$out" \
    --batch-size 256 --max-seq-len 128 --workers 2 \
    --single-muatm-only --eval-only --eval-split test

"$python" compare_checkpoint_paired.py \
    --before runs/mixed_universal/continue_5k_stochastic_p/test_best2500/test_residuals.npz \
    --after "$out/test_residuals.npz" \
    --masks runs/mixed_universal/masks_v2.npz \
    --pred-data "$pred_data" --pred-anchors all_predsignal_p070_track_anchors_v1.h5 \
    --gt-data "$gt_data" --output "$out/paired_v4_vs_trimmed.json"
echo "trimmed_test_complete $(date -u +%FT%TZ)"
