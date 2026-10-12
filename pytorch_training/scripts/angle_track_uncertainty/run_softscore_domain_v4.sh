#!/usr/bin/env bash
set -euo pipefail

cd /home/plotnikovgp/tmp/angle_reconstruction/track_anchor
python=/home/plotnikovgp/tmp/ENTER/bin/python3.11
root=runs/mixed_universal/continue_v4_softscore_column_2k
output=$root/domain_p070
checkpoint=$root/best.pt
mc_data=/home/plotnikovgp/tmp/angle_reconstruction/data/baikal_2020_all_predsignal_p070_2s8h.h5
exp_data=/home/plotnikovgp/tmp/angle_reconstruction/data/baikal_exp_predsignal_p070_2s8h_v2audit.h5
norm_data=/home/plotnikovgp/tmp/angle_reconstruction/data/baikal_mc_merged2020_signal_8-2_angl_norm.h5
before=runs/mixed_universal/threshold_sweep_v4

"$python" infer_softscore_for_audit.py \
    --checkpoint "$checkpoint" \
    --angle-checkpoint nue2_best_q68_trim_cap8.pt \
    --track-checkpoint runs/direct_line_from_best/best_line_q68.pt \
    --norm-data "$norm_data" --data "$exp_data" --split test \
    --threshold 0.70 --output "$output/exp_test.csv"
"$python" infer_softscore_for_audit.py \
    --checkpoint "$checkpoint" \
    --angle-checkpoint nue2_best_q68_trim_cap8.pt \
    --track-checkpoint runs/direct_line_from_best/best_line_q68.pt \
    --norm-data "$norm_data" --data "$mc_data" --split val \
    --max-events 25972 --threshold 0.70 --output "$output/muatm_val.csv"

"$python" evaluate_mc_angle_by_threshold.py \
    --predictions "$output/muatm_val.csv" --mc-data "$mc_data" --split val \
    --p070-anchors all_predsignal_p070_track_anchors_v1.h5 \
    --output "$output/mc_quality.json"
"$python" compare_mc_quality_checkpoints.py \
    --before "$before/mc_quality_p070.npz" --after "$output/mc_quality.npz" \
    --output "$output/paired_mc_quality.json"
"$python" compare_domain_models_paired.py \
    --before-exp "$before/exp_test_p070.csv" --after-exp "$output/exp_test.csv" \
    --before-mc "$before/muatm_val_p070.csv" --after-mc "$output/muatm_val.csv" \
    --output-dir "$output" --bootstrap 500
echo "softscore_domain_complete $(date -u +%FT%TZ)"
