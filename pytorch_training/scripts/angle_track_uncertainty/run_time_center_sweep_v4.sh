#!/usr/bin/env bash
set -euo pipefail

cd /home/plotnikovgp/tmp/angle_reconstruction/track_anchor
python=/home/plotnikovgp/tmp/ENTER/bin/python3.11
bundle=runs/mixed_universal/universal_angle_track_uncertainty_v4.pt
output=runs/time_center_sweep_v4
mkdir -p "$output"

for pair in 000:0.0 050:0.5 100:1.0; do
    tag=${pair%%:*}
    alpha=${pair#*:}
    "$python" infer_rethresholded_for_audit.py \
        --bundle "$bundle" \
        --data /home/plotnikovgp/tmp/angle_reconstruction/data/baikal_exp_predsignal_p070_2s8h_v2audit.h5 \
        --split test --threshold 0.70 --event-stride 4 \
        --center-time-alpha "$alpha" --output "$output/exp_a${tag}.csv"
    "$python" infer_rethresholded_for_audit.py \
        --bundle "$bundle" \
        --data /home/plotnikovgp/tmp/angle_reconstruction/data/baikal_2020_all_predsignal_p070_2s8h.h5 \
        --split val --max-events 25972 --threshold 0.70 --event-stride 4 \
        --center-time-alpha "$alpha" --output "$output/muatm_a${tag}.csv"
done
"$python" summarize_time_center_sweep.py \
    --sweep-dir "$output" \
    --mc-data /home/plotnikovgp/tmp/angle_reconstruction/data/baikal_2020_all_predsignal_p070_2s8h.h5 \
    --p070-anchors all_predsignal_p070_track_anchors_v1.h5
