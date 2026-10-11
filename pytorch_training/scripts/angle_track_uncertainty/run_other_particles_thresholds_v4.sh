#!/usr/bin/env bash
set -euo pipefail

cd /home/plotnikovgp/tmp/angle_reconstruction/track_anchor

python=/home/plotnikovgp/tmp/ENTER/bin/python3.11
bundle=runs/mixed_universal/universal_angle_track_uncertainty_v4.pt
mc_data=/home/plotnikovgp/tmp/angle_reconstruction/data/baikal_2020_all_predsignal_p070_2s8h.h5
anchors=all_predsignal_p070_track_anchors_v1.h5
output=runs/mixed_universal/threshold_sweep_v4/other_particles

# The validation HDF5 is contiguous by type. Evaluate all 4,185 nuatm rows,
# and every fourth of 21,744 nue2 rows so the sample spans all source runs.
for pair in 070:0.70 075:0.75 080:0.80 090:0.90; do
    tag=${pair%%:*}
    threshold=${pair#*:}
    echo "BEGIN threshold=$threshold $(date -u +%FT%TZ)"
    for particle in nuatm nue2; do
        if [ "$particle" = nuatm ]; then
            start=25972
            count=4185
            stride=1
        else
            start=30157
            count=21744
            stride=4
        fi
        csv=$output/${particle}_val_p${tag}.csv
        if [ ! -e "$csv" ]; then
            "$python" infer_rethresholded_for_audit.py \
                --bundle "$bundle" --data "$mc_data" --split val \
                --start-event "$start" --max-events "$count" \
                --event-stride "$stride" --threshold "$threshold" --output "$csv"
        fi
        "$python" evaluate_mc_angle_by_threshold.py \
            --predictions "$csv" --mc-data "$mc_data" --split val \
            --p070-anchors "$anchors" \
            --output "$output/${particle}_quality_p${tag}.json"
    done
    echo "DONE threshold=$threshold $(date -u +%FT%TZ)"
done

"$python" summarize_other_particles_thresholds_v4.py --sweep-dir "$output"
