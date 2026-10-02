#!/usr/bin/env bash
# Masked pseudo-label sweep: confident noise (P < p_lo) = 0, confident signal (P > p_hi) = 1,
# ambiguous hits masked from loss. Three cells: (0.1/0.9), (0.2/0.8), (0.3/0.7).
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJ_ROOT"

export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
NUM_WORKERS="${NUM_WORKERS:-2}"
N_ROUNDS="${N_ROUNDS:-3}"
EVENTS_PER_ROUND="${EVENTS_PER_ROUND:-3000000}"
BATCH="${BATCH:-128}"
VAL_EVERY="${VAL_EVERY:-500}"
DIAG_AFTER_CUT_TARGET="${DIAG_AFTER_CUT_TARGET:-8000}"

BASE_CKPT="${BASE_CKPT:-$PROJ_ROOT/checkpoints/noise_sig_experiments/k_nsol_labelneq0_hs128/best.ckpt}"
EXP_PROJECT="${EXP_PROJECT:-noise_sig_experiments}"

DIAG="$PROJ_ROOT/scripts/iter_pseudolabel/diagnostics.py"
MAKE_MASKED="$PROJ_ROOT/scripts/iter_pseudolabel/make_masked_pseudolabels.py"

PL_ROOT="${PL_ROOT:-$PROJ_ROOT/data/pseudolabels}"
PLOTS_ROOT="${PLOTS_ROOT:-$PROJ_ROOT/plots/iter_pseudolabel}"
CKPT_ROOT="${CKPT_ROOT:-$PROJ_ROOT/checkpoints/$EXP_PROJECT}"

mkdir -p "$PL_ROOT" "$PLOTS_ROOT"

log() { echo -e "\n=== $* ===" >&2; }

run_masked_cell() {
    local name="$1" p_lo="$2" p_hi="$3"
    log "Cell: $name (p_lo=$p_lo, p_hi=$p_hi)"

    local ckpt="$BASE_CKPT"
    for k in $(seq 1 "$N_ROUNDS"); do
        local pl_dir="$PL_ROOT/$name"
        mkdir -p "$pl_dir"
        local pl="$pl_dir/iter_$k.h5"

        if [[ -f "$pl" ]]; then
            log "[$name] round $k :: pseudo-labels exist, skipping"
        else
            log "[$name] round $k :: generate masked pseudo-labels with $ckpt"
            python "$MAKE_MASKED" --ckpt "$ckpt" --p-lo "$p_lo" --p-hi "$p_hi" --out "$pl"
        fi

        local run_name="pseudo_${name}_iter_$k"
        local done_marker="$CKPT_ROOT/$run_name/best_exp_data.ckpt"
        if [[ ! -f "$done_marker" ]]; then
            log "[$name] round $k :: train ($EVENTS_PER_ROUND events) -> $run_name"
            python train.py +experiment=pseudolabel_masked \
                exp_project="$EXP_PROJECT" \
                exp_name="$run_name" \
                from_checkpoint="$ckpt" \
                data.datasets.1.labels_path="$pl" \
                data.weights="[0.5,0.5]" \
                data.num_workers="$NUM_WORKERS" \
                training.batch_size="$BATCH" \
                training.num_train_steps_per_validation="$VAL_EVERY" \
                training.events_per_round="$EVENTS_PER_ROUND" \
                training.warmup_steps=1 \
                use_clearml="${USE_CLEARML:-false}"
        else
            log "[$name] round $k :: training done ($done_marker exists), skipping"
        fi

        local new_ckpt="$CKPT_ROOT/$run_name/best_exp_data.ckpt"
        [[ ! -f "$new_ckpt" ]] && new_ckpt="$CKPT_ROOT/$run_name/best_mc_2020.ckpt"
        [[ ! -f "$new_ckpt" ]] && new_ckpt="$CKPT_ROOT/$run_name/best.ckpt"
        if [[ ! -f "$new_ckpt" ]]; then
            log "[$name] ERROR: no checkpoint found in $CKPT_ROOT/$run_name; stopping cell"
            return 1
        fi
        ckpt="$new_ckpt"

        local diag_out="$PLOTS_ROOT/$name/iter_$k"
        if [[ -f "$diag_out/summary.json" ]]; then
            log "[$name] round $k :: diagnostics exist, skipping"
        else
            log "[$name] round $k :: diagnostics on $new_ckpt"
            python "$DIAG" --ckpt "$new_ckpt" \
                --out-dir "$diag_out" \
                --after-cut-target "$DIAG_AFTER_CUT_TARGET" \
                --ckpt-label "$run_name"
        fi
    done
}

ONLY_CELL="${ONLY_CELL:-}"

CELLS=(
  "masked_0p1_0p9 0.1 0.9"
  "masked_0p2_0p8 0.2 0.8"
  "masked_0p3_0p7 0.3 0.7"
)

for entry in "${CELLS[@]}"; do
    set -- $entry
    name="$1" p_lo="$2" p_hi="$3"
    if [[ -n "$ONLY_CELL" && "$name" != "$ONLY_CELL" ]]; then continue; fi
    run_masked_cell "$name" "$p_lo" "$p_hi"
done

log "All masked cells finished."
