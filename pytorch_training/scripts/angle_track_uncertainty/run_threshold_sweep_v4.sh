#!/usr/bin/env bash
set -euo pipefail

cd /home/plotnikovgp/tmp/angle_reconstruction/track_anchor

python=/home/plotnikovgp/tmp/ENTER/bin/python3.11
bundle=runs/mixed_universal/universal_angle_track_uncertainty_v4.pt
exp_data=/home/plotnikovgp/tmp/angle_reconstruction/data/baikal_exp_predsignal_p070_2s8h_v2audit.h5
mc_data=/home/plotnikovgp/tmp/angle_reconstruction/data/baikal_2020_all_predsignal_p070_2s8h.h5
anchors=all_predsignal_p070_track_anchors_v1.h5
output=runs/mixed_universal/threshold_sweep_v4

for pair in 070:0.70 075:0.75 080:0.80 085:0.85 090:0.90; do
    tag=${pair%%:*}
    threshold=${pair#*:}
    exp_csv=$output/exp_test_p${tag}.csv
    mc_csv=$output/muatm_val_p${tag}.csv
    echo "BEGIN threshold=$threshold $(date -u +%FT%TZ)"
    "$python" infer_rethresholded_for_audit.py \
        --bundle "$bundle" --data "$exp_data" --split test \
        --threshold "$threshold" --output "$exp_csv"
    "$python" infer_rethresholded_for_audit.py \
        --bundle "$bundle" --data "$mc_data" --split val \
        --max-events 25972 --threshold "$threshold" --output "$mc_csv"
    "$python" evaluate_mc_angle_by_threshold.py \
        --predictions "$mc_csv" --mc-data "$mc_data" --split val \
        --p070-anchors "$anchors" --output "$output/mc_quality_p${tag}.json"
    "$python" analyze_domain_predictions.py \
        --exp-csv "$exp_csv" --mc-csv "$mc_csv" \
        --p070-anchors "$anchors" --threshold "$threshold" \
        --model-label v4 --output-dir "$output/domain_p${tag}"
    echo "DONE threshold=$threshold $(date -u +%FT%TZ)"
done

"$python" compare_domain_threshold_common_v4.py \
    --sweep-dir "$output" --bootstrap 300
"$python" summarize_threshold_sweep_v4.py --sweep-dir "$output"
