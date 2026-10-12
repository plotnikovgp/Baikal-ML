#!/usr/bin/env bash
set -euo pipefail

weight=${1:?Usage: run_clean_paired_audit_v4.sh 0.1|0.3 GPU}
gpu=${2:-0}
case "$weight" in 0.1|0.3) ;; *) echo "Unsupported clean-paired weight: $weight" >&2; exit 2 ;; esac
cd /home/plotnikovgp/tmp/angle_reconstruction/track_anchor

python=/home/plotnikovgp/tmp/ENTER/bin/python3.11
root="runs/mixed_universal/continue_v4_clean_paired_w${weight}_3k"
checkpoint="$root/best.pt"
test_out="$root/test_best"
domain_out="$root/domain_p070"
pred_data=/home/plotnikovgp/tmp/angle_reconstruction/data/baikal_2020_all_predsignal_p070_2s8h.h5
gt_data=/home/plotnikovgp/tmp/angle_reconstruction/data/baikal_mc_merged2020_signal_8-2_angl_norm.h5
exp_data=/home/plotnikovgp/tmp/angle_reconstruction/data/baikal_exp_predsignal_p070_2s8h_v2audit.h5
before=runs/mixed_universal/threshold_sweep_v4

test -f "$checkpoint" || { echo "No validation-selected candidate: $checkpoint" >&2; exit 2; }

CUDA_VISIBLE_DEVICES="$gpu" "$python" train_mixed_universal.py \
    --pred-data "$pred_data" --norm-data "$gt_data" \
    --pred-anchors all_predsignal_p070_track_anchors_v1.h5 \
    --gt-data "$gt_data" --gt-anchors nue2_track_anchors_v3.h5 \
    --masks runs/mixed_universal/masks_v2.npz \
    --angle-checkpoint nue2_best_q68_trim_cap8.pt \
    --track-checkpoint runs/direct_line_from_best/best_line_q68.pt \
    --load-model "$checkpoint" --output-dir "$test_out" \
    --batch-size 256 --max-seq-len 128 --workers 2 \
    --single-muatm-only --eval-only --eval-split test

"$python" compare_checkpoint_paired.py \
    --before runs/mixed_universal/continue_5k_stochastic_p/test_best2500/test_residuals.npz \
    --after "$test_out/test_residuals.npz" \
    --masks runs/mixed_universal/masks_v2.npz \
    --pred-data "$pred_data" --pred-anchors all_predsignal_p070_track_anchors_v1.h5 \
    --gt-data "$gt_data" --output "$test_out/paired_v4_vs_clean_paired.json"

"$python" infer_softscore_for_audit.py \
    --checkpoint "$checkpoint" \
    --angle-checkpoint nue2_best_q68_trim_cap8.pt \
    --track-checkpoint runs/direct_line_from_best/best_line_q68.pt \
    --norm-data "$gt_data" --data "$exp_data" --split test \
    --threshold 0.70 --output "$domain_out/exp_test.csv"
"$python" infer_softscore_for_audit.py \
    --checkpoint "$checkpoint" \
    --angle-checkpoint nue2_best_q68_trim_cap8.pt \
    --track-checkpoint runs/direct_line_from_best/best_line_q68.pt \
    --norm-data "$gt_data" --data "$pred_data" --split val \
    --max-events 25972 --threshold 0.70 --output "$domain_out/muatm_val.csv"
"$python" evaluate_mc_angle_by_threshold.py \
    --predictions "$domain_out/muatm_val.csv" --mc-data "$pred_data" --split val \
    --p070-anchors all_predsignal_p070_track_anchors_v1.h5 \
    --output "$domain_out/mc_quality.json"
"$python" compare_mc_quality_checkpoints.py \
    --before "$before/mc_quality_p070.npz" --after "$domain_out/mc_quality.npz" \
    --output "$domain_out/paired_mc_quality.json"
"$python" compare_domain_models_paired.py \
    --before-exp "$before/exp_test_p070.csv" --after-exp "$domain_out/exp_test.csv" \
    --before-mc "$before/muatm_val_p070.csv" --after-mc "$domain_out/muatm_val.csv" \
    --output-dir "$domain_out" --bootstrap 500
echo "clean_paired_audit_complete $weight $(date -u +%FT%TZ)"
