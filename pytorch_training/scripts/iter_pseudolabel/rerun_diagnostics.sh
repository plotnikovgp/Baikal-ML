#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HDF5_USE_FILE_LOCKING=FALSE

ROOT="/home/plotnikovgp/baikal/Baikal-ML/pytorch_training"
CKPT_ROOT="$ROOT/checkpoints/noise_sig_experiments"
PLOT_ROOT="$ROOT/plots/iter_pseudolabel"
DIAG="$ROOT/scripts/iter_pseudolabel/diagnostics.py"

run_diag() {
    local ckpt="$1" out_dir="$2" label="$3"
    if [[ ! -f "$ckpt" ]]; then
        echo "SKIP (no checkpoint): $ckpt"
        return
    fi
    if [[ -f "$out_dir/ks_signal_hits.json" ]]; then
        echo "SKIP (already has new metrics): $label"
        return
    fi
    echo "=== DIAG: $label ==="
    echo "  ckpt=$ckpt"
    echo "  out=$out_dir"
    python3 "$DIAG" \
        --ckpt "$ckpt" \
        --out-dir "$out_dir" \
        --ckpt-label "$label" \
        --n-events 15000 \
        --after-cut-target 8000 \
        --max-indices-scan 2000000 \
        --batch-size 128
    echo "  DONE: $label"
    echo ""
}

# Base checkpoint
run_diag \
    "$CKPT_ROOT/k_nsol_labelneq0_hs128/best.ckpt" \
    "$PLOT_ROOT/round_0_base" \
    "k_nsol_labelneq0_hs128 (base)"

# Soft cells from main sweep
for cell in soft_exp0p50_mc0p50 soft_exp0p70_mc0p25 soft_exp0p25_mc0p75 soft_exp1p00_mc0p00 soft_exp0p10_mc0p90; do
    for r in 1 2 3; do
        ckpt_name="pseudo_${cell}_iter_${r}"
        ckpt="$CKPT_ROOT/$ckpt_name/best_exp_data.ckpt"
        [[ ! -f "$ckpt" ]] && ckpt="$CKPT_ROOT/$ckpt_name/best_mc_2020.ckpt"
        [[ ! -f "$ckpt" ]] && ckpt="$CKPT_ROOT/$ckpt_name/best.ckpt"
        run_diag "$ckpt" "$PLOT_ROOT/$cell/iter_${r}" "${cell} iter ${r}"
    done
done

# Preview LR run
for r in 1 2 3 4 5; do
    ckpt_name="pseudo_soft_50_50_lr5e4_iter_${r}"
    ckpt="$CKPT_ROOT/$ckpt_name/best_exp_data.ckpt"
    [[ ! -f "$ckpt" ]] && ckpt="$CKPT_ROOT/$ckpt_name/best_mc_2020.ckpt"
    [[ ! -f "$ckpt" ]] && ckpt="$CKPT_ROOT/$ckpt_name/best.ckpt"
    run_diag "$ckpt" "$PLOT_ROOT/soft_50_50_lr5e4/iter_${r}" "soft_50_50_lr5e4 iter ${r}"
done

# Extra reference checkpoints (different architectures, auto-detected)
run_diag \
    "$CKPT_ROOT/k_nsol_labelneq0_da_hs128_k0p0001/best_mc_2020.ckpt" \
    "$PLOT_ROOT/da_k0p0001" \
    "DA k=0.0001"

run_diag \
    "$CKPT_ROOT/noise_sig_tres_abs_merged/best.ckpt" \
    "$PLOT_ROOT/tres_abs_merged" \
    "tres_abs_merged"

run_diag \
    "$CKPT_ROOT/k_nsol_labelneq0_hs128_zm/best.ckpt" \
    "$PLOT_ROOT/zm_hs128" \
    "zm hs128"

echo "ALL DIAGNOSTICS COMPLETE"
