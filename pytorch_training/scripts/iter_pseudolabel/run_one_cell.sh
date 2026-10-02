#!/usr/bin/env bash
# Run one pseudo-label cell with overridable LR / batch / rounds.
#
# Required env vars: CELL_NAME, FAMILY (soft|hard), WEIGHTS (Hydra list, e.g. "[0.5,0.5]")
# Optional:          THRESHOLD (only for hard), LR, BATCH, N_ROUNDS,
#                    NUM_WORKERS, EVENTS_PER_ROUND, VAL_EVERY,
#                    BASE_CKPT, DIAG_AFTER_CUT_TARGET, USE_CLEARML
#
# Example:
#   CELL_NAME=soft_50_50_lr5e4 FAMILY=soft WEIGHTS='[0.5,0.5]' \
#   LR=5e-4 N_ROUNDS=5 BATCH=64 CUDA_VISIBLE_DEVICES=0 \
#       bash scripts/iter_pseudolabel/run_one_cell.sh > logs/preview_lr5e4.log 2>&1

set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJ_ROOT"

export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

NUM_WORKERS="${NUM_WORKERS:-2}"
N_ROUNDS="${N_ROUNDS:-3}"
EVENTS_PER_ROUND="${EVENTS_PER_ROUND:-3000000}"
BATCH="${BATCH:-128}"
VAL_EVERY="${VAL_EVERY:-500}"
LR="${LR:-1e-4}"
WARMUP="${WARMUP:-200}"
DIAG_AFTER_CUT_TARGET="${DIAG_AFTER_CUT_TARGET:-8000}"
USE_CLEARML="${USE_CLEARML:-false}"

BASE_CKPT="${BASE_CKPT:-$PROJ_ROOT/checkpoints/noise_sig_experiments/k_nsol_labelneq0_hs128/best.ckpt}"
EXP_PROJECT="${EXP_PROJECT:-noise_sig_experiments}"

DIAG="$PROJ_ROOT/scripts/iter_pseudolabel/diagnostics.py"
MAKE_SOFT="$PROJ_ROOT/scripts/iter_pseudolabel/make_soft_pseudolabels.py"
MAKE_HARD="$PROJ_ROOT/scripts/iter_pseudolabel/make_hard_pseudolabels.py"

PL_ROOT="$PROJ_ROOT/data/pseudolabels"
PLOTS_ROOT="$PROJ_ROOT/plots/iter_pseudolabel"
CKPT_ROOT="$PROJ_ROOT/checkpoints/$EXP_PROJECT"

: "${CELL_NAME:?CELL_NAME is required}"
: "${FAMILY:?FAMILY is required (soft|hard)}"
: "${WEIGHTS:?WEIGHTS is required, e.g. [0.5,0.5]}"
if [[ "$FAMILY" == "hard" ]]; then
    : "${THRESHOLD:?THRESHOLD is required for hard cells}"
fi

log() { echo -e "\n=== $* ===" >&2; }

ckpt="$BASE_CKPT"
for k in $(seq 1 "$N_ROUNDS"); do
    pl_dir="$PL_ROOT/$CELL_NAME"
    mkdir -p "$pl_dir"
    pl="$pl_dir/iter_$k.h5"

    if [[ -f "$pl" ]]; then
        log "[$CELL_NAME] round $k :: pseudo-labels exist, skipping"
    else
        log "[$CELL_NAME] round $k :: generate pseudo-labels with $ckpt"
        if [[ "$FAMILY" == "soft" ]]; then
            python "$MAKE_SOFT" --ckpt "$ckpt" --out "$pl"
        else
            python "$MAKE_HARD" --ckpt "$ckpt" --threshold "$THRESHOLD" --out "$pl"
        fi
    fi

    run_name="pseudo_${CELL_NAME}_iter_$k"
    done_marker="$CKPT_ROOT/$run_name/best_exp_data.ckpt"
    if [[ -f "$done_marker" ]]; then
        log "[$CELL_NAME] round $k :: training done, skipping"
    else
        log "[$CELL_NAME] round $k :: train ($EVENTS_PER_ROUND events, lr=$LR, bs=$BATCH) -> $run_name"
        if [[ "$FAMILY" == "soft" ]]; then
            python train.py +experiment=pseudolabel_soft \
                exp_name="$run_name" \
                from_checkpoint="$ckpt" \
                data.datasets.1.labels_path="$pl" \
                data.weights="$WEIGHTS" \
                data.num_workers="$NUM_WORKERS" \
                training.lr="$LR" \
                training.warmup_steps="$WARMUP" \
                training.batch_size="$BATCH" \
                training.num_train_steps_per_validation="$VAL_EVERY" \
                training.events_per_round="$EVENTS_PER_ROUND" \
                use_clearml="$USE_CLEARML"
        else
            python train.py +experiment=pseudolabel_hard \
                exp_name="$run_name" \
                from_checkpoint="$ckpt" \
                data.datasets.1.labels_path="$pl" \
                data.weights="$WEIGHTS" \
                data.num_workers="$NUM_WORKERS" \
                training.lr="$LR" \
                training.warmup_steps="$WARMUP" \
                training.batch_size="$BATCH" \
                training.num_train_steps_per_validation="$VAL_EVERY" \
                training.events_per_round="$EVENTS_PER_ROUND" \
                use_clearml="$USE_CLEARML"
        fi
    fi

    new_ckpt="$CKPT_ROOT/$run_name/best_exp_data.ckpt"
    [[ -f "$new_ckpt" ]] || new_ckpt="$CKPT_ROOT/$run_name/best_mc_2020.ckpt"
    [[ -f "$new_ckpt" ]] || { echo "no ckpt found"; exit 1; }
    ckpt="$new_ckpt"

    diag_out="$PLOTS_ROOT/$CELL_NAME/iter_$k"
    if [[ -f "$diag_out/summary.json" ]]; then
        log "[$CELL_NAME] round $k :: diagnostics exist, skipping"
    else
        log "[$CELL_NAME] round $k :: diagnostics on $new_ckpt"
        python "$DIAG" --ckpt "$new_ckpt" \
            --out-dir "$diag_out" \
            --after-cut-target "$DIAG_AFTER_CUT_TARGET" \
            --ckpt-label "$run_name"
    fi
done

log "[$CELL_NAME] done"
