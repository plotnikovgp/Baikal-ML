#!/usr/bin/env bash
set -euo pipefail

cd /home/plotnikovgp/tmp/angle_reconstruction/track_anchor
python=/home/plotnikovgp/tmp/ENTER/bin/python3.11
root=runs/mixed_universal/threshold_sweep_v4
out=runs/mixed_universal/physics_fit_v1
mc=/home/plotnikovgp/tmp/angle_reconstruction/data/baikal_2020_all_predsignal_p070_2s8h.h5
exp=/home/plotnikovgp/tmp/angle_reconstruction/data/baikal_exp_predsignal_p070_2s8h_v2audit.h5
anchors=all_predsignal_p070_track_anchors_v1.h5

"$python" evaluate_spacetime_direction.py \
    --predictions "$root/muatm_val_p070.csv" --data "$mc" --split val \
    --anchors "$anchors" --output-prefix "$out/muatm_val"
"$python" evaluate_spacetime_direction.py \
    --predictions "$root/other_particles/nuatm_val_p070.csv" \
    --data "$mc" --split val --anchors "$anchors" \
    --output-prefix "$out/nuatm_val"
"$python" evaluate_spacetime_direction.py \
    --predictions "$root/other_particles/nue2_val_p070.csv" \
    --data "$mc" --split val --anchors "$anchors" \
    --output-prefix "$out/nue2_val_sample"
"$python" evaluate_spacetime_direction.py \
    --predictions "$root/exp_test_p070.csv" --data "$exp" --split test \
    --output-prefix "$out/exp_test"
echo "physics_fit_v1_complete $(date -u +%FT%TZ)"
