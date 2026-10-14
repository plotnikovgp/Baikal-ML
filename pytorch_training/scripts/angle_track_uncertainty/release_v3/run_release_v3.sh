#!/usr/bin/env bash
# Raw-HDF5 -> frozen signal/noise -> one angle/line model, all on CPU.
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 RAW_EVENTS.h5 OUTPUT_DIRECTORY" >&2
    exit 2
fi

raw_h5=$1
output_dir=$2
python_bin=${PYTHON_BIN:-python3}
release_dir=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "$release_dir/../../../.." && pwd)
mkdir -p "$output_dir"
builder_extra=()
if [ -n "${MAX_KEPT_TEST:-}" ]; then
    builder_extra=(--max-kept-test "$MAX_KEPT_TEST")
fi

"$python_bin" "$repo_root/pytorch_training/scripts/build_predicted_signal_dataset.py" \
    --source "$raw_h5" --splits test \
    --output "$output_dir/selected_p050.h5" \
    --checkpoint "$release_dir/signal_noise.ckpt" \
    --model-norm-source "$release_dir/classifier_norm.h5" \
    --output-norm-source "$release_dir/angle_norm.h5" \
    --output-norm-group norm_param \
    --threshold 0.50 --particle all --min-hits 8 --min-strings 2 \
    --device cpu "${builder_extra[@]}"

"$python_bin" "$repo_root/pytorch_training/scripts/angle_track_uncertainty/infer_rethresholded_for_audit.py" \
    --bundle "$release_dir/angle_track_uncertainty.pt" \
    --data "$output_dir/selected_p050.h5" --split test --threshold 0.70 \
    --output "$output_dir/predictions_calibrated_p070.csv" \
    --device cpu --threads 8

"$python_bin" "$repo_root/pytorch_training/scripts/angle_track_uncertainty/infer_rethresholded_for_audit.py" \
    --bundle "$release_dir/angle_track_uncertainty.pt" \
    --data "$output_dir/selected_p050.h5" --split test --threshold 0.70 \
    --conditional-low-threshold 0.50 --gate-theta-max-deg 90 \
    --max-added-photon-residual-ns 200 \
    --output "$output_dir/predictions_best_quality_gate200.csv" \
    --device cpu --threads 8
