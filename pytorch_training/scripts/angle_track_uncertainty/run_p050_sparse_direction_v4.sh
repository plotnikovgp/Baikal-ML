#!/usr/bin/env bash
set -euo pipefail

steps=${1:-2000}
gpu=${2:-0}
extra_weight=${3:-0.30}
seed=${4:-211}

work=/home/plotnikovgp/tmp/angle_reconstruction/track_anchor
script=/home/plotnikovgp/tmp/baikal-angle-uncertainty-commit/pytorch_training/scripts/angle_track_uncertainty/train_mixed_universal.py
output="$work/runs/mixed_universal/p050_sparse_dir_noprimary_w${extra_weight}_${steps}steps_seed${seed}"
if [[ -e "$output" ]]; then
    echo "Refusing to reuse output: $output" >&2
    exit 2
fi
val_every=250
if (( steps < val_every )); then
    val_every=$steps
fi

cd "$work"
exec env CUDA_VISIBLE_DEVICES="$gpu" /home/plotnikovgp/tmp/ENTER/bin/python3.11 "$script" \
    --pred-data /home/plotnikovgp/tmp/angle_reconstruction/data/baikal_2020_all_predsignal_p070_2s8h.h5 \
    --norm-data /home/plotnikovgp/tmp/angle_reconstruction/data/baikal_mc_merged2020_signal_8-2_angl_norm.h5 \
    --pred-anchors "$work/all_predsignal_p070_track_anchors_v1.h5" \
    --extra-pred-data /home/plotnikovgp/tmp/angle_reconstruction/data/baikal_2020_all_predsignal_p050_train250k.h5 \
    --extra-pred-anchors "$work/all_predsignal_p050_train250k_overlap_anchors_v1.h5" \
    --extra-pred-weight "$extra_weight" --extra-pred-sparse-direction-only \
    --gt-data /home/plotnikovgp/tmp/angle_reconstruction/data/baikal_mc_merged2020_signal_8-2_angl_norm.h5 \
    --gt-anchors "$work/nue2_track_anchors_v3.h5" \
    --masks "$work/runs/mixed_universal/masks_v2.npz" \
    --angle-checkpoint "$work/nue2_best_q68_trim_cap8.pt" \
    --track-checkpoint "$work/runs/direct_line_from_best/best_line_q68.pt" \
    --load-model "$work/runs/mixed_universal/continue_5k_stochastic_p/best.pt" \
    --output-dir "$output" \
    --batch-size 256 --max-seq-len 128 --workers 2 \
    --steps "$steps" --val-every "$val_every" \
    --gt-weight 0.4 --point-weight 0.5 \
    --backbone-lr 5e-6 --point-lr 1e-5 \
    --max-pred-q68 4.9 --max-gt-q68 4.4 \
    --pred-score-weight 0.5 --muatm-score-weight 0.03 \
    --short-score-weight 0.5 \
    --single-muatm-only \
    --pred-thinning-prob 0.5 --pred-threshold-min 0.7 --pred-threshold-max 0.9 \
    --seed "$seed"
