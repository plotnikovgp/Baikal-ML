#!/usr/bin/env bash
# Iterative pseudo-label sweep on Baikal EXP data.
#
# Two families:
#   - soft:  raw P(signal) targets, swept over 5 (EXP, MC) mix ratios
#   - hard:  binary pseudo-labels, swept over 3 thresholds at fixed 50/50 mix
#
# Per cell: N_ROUNDS rounds of (pseudo-label -> train EVENTS_PER_ROUND events -> diagnose).
# Validation always on MC val (baikal_2020_balanced_val_2k_each.h5).

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
MAKE_SOFT="$PROJ_ROOT/scripts/iter_pseudolabel/make_soft_pseudolabels.py"
MAKE_HARD="$PROJ_ROOT/scripts/iter_pseudolabel/make_hard_pseudolabels.py"

PL_ROOT="$PROJ_ROOT/data/pseudolabels"
PLOTS_ROOT="$PROJ_ROOT/plots/iter_pseudolabel"
CKPT_ROOT="$PROJ_ROOT/checkpoints/$EXP_PROJECT"

mkdir -p "$PL_ROOT" "$PLOTS_ROOT"

log() { echo -e "\n=== $* ===" >&2; }

# Round-0 baseline (shared across all cells)
if [[ ! -f "$PLOTS_ROOT/round_0_base/summary.json" ]]; then
  log "Round-0 baseline diagnostics on $BASE_CKPT"
  python "$DIAG" --ckpt "$BASE_CKPT" \
      --out-dir "$PLOTS_ROOT/round_0_base" \
      --after-cut-target "$DIAG_AFTER_CUT_TARGET" \
      --ckpt-label "round_0_base"
fi

run_cell() {
    # Args:
    #   $1 = cell name
    #   $2 = family: soft|hard
    #   $3 = mix weight expression (e.g. "[0.5,0.5]") or "MC_ONLY"
    #   $4 = (only for hard) threshold
    local name="$1"
    local family="$2"
    local weights="$3"
    local thr="${4:-}"

    log "Cell: $name (family=$family, weights=$weights, thr=${thr:-N/A})"

    local ckpt="$BASE_CKPT"
    for k in $(seq 1 "$N_ROUNDS"); do
        local pl_dir="$PL_ROOT/$name"
        mkdir -p "$pl_dir"
        local pl="$pl_dir/iter_$k.h5"

        if [[ -f "$pl" ]]; then
            log "[$name] round $k :: pseudo-labels already exist at $pl, skipping generation"
        else
            log "[$name] round $k :: generate pseudo-labels with $ckpt"
            if [[ "$family" == "soft" ]]; then
                python "$MAKE_SOFT" --ckpt "$ckpt" --out "$pl"
            else
                python "$MAKE_HARD" --ckpt "$ckpt" --threshold "$thr" --out "$pl"
            fi
        fi

        local run_name="pseudo_${name}_iter_$k"
        local done_marker="$CKPT_ROOT/$run_name/best_exp_data.ckpt"
        if [[ -f "$done_marker" ]]; then
            log "[$name] round $k :: training already complete ($done_marker exists), skipping"
        else
            log "[$name] round $k :: train ($EVENTS_PER_ROUND events) -> $run_name"
            true # placeholder to keep the rest of the if/else cleanly
        fi
        if [[ ! -f "$done_marker" ]]; then
        if [[ "$family" == "soft" ]]; then
            python train.py +experiment=pseudolabel_soft \
                exp_name="$run_name" \
                from_checkpoint="$ckpt" \
                data.datasets.1.labels_path="$pl" \
                data.weights="$weights" \
                data.num_workers="$NUM_WORKERS" \
                training.batch_size="$BATCH" \
                training.num_train_steps_per_validation="$VAL_EVERY" \
                training.events_per_round="$EVENTS_PER_ROUND" \
                use_clearml="${USE_CLEARML:-false}"
        else
            python train.py +experiment=pseudolabel_hard \
                exp_name="$run_name" \
                from_checkpoint="$ckpt" \
                data.datasets.1.labels_path="$pl" \
                data.weights="$weights" \
                data.num_workers="$NUM_WORKERS" \
                training.batch_size="$BATCH" \
                training.num_train_steps_per_validation="$VAL_EVERY" \
                training.events_per_round="$EVENTS_PER_ROUND" \
                use_clearml="${USE_CLEARML:-false}"
        fi
        fi  # end skip-if-done

        # Iterative pseudo-labeling: chain the *EXP-adapted* checkpoint, since training
        # to fit our own EXP pseudo-labels typically increases MC val loss slightly, so
        # best_mc_2020.ckpt would just be the unchanged base ckpt (step 0).
        # `best_exp_data.ckpt` is the model state with lowest BCE on the EXP pseudo-targets,
        # which is the natural "advanced" checkpoint for the next pseudo-label generation.
        local new_ckpt="$CKPT_ROOT/$run_name/best_exp_data.ckpt"
        if [[ ! -f "$new_ckpt" ]]; then
            new_ckpt="$CKPT_ROOT/$run_name/best_mc_2020.ckpt"
        fi
        if [[ ! -f "$new_ckpt" ]]; then
            new_ckpt="$CKPT_ROOT/$run_name/best.ckpt"
        fi
        if [[ ! -f "$new_ckpt" ]]; then
            log "[$name] ERROR: expected ckpt not found in $CKPT_ROOT/$run_name; stopping cell"
            return 1
        fi
        ckpt="$new_ckpt"

        local diag_out="$PLOTS_ROOT/$name/iter_$k"
        if [[ -f "$diag_out/summary.json" ]]; then
            log "[$name] round $k :: diagnostics already exist, skipping"
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

# Cell name | EXP weight | MC weight (Hydra list literal)
SOFT_CELLS=(
  "soft_exp1p00_mc0p00 [1.0,0.0]"
  "soft_exp0p70_mc0p25 [0.7,0.25]"
  "soft_exp0p50_mc0p50 [0.5,0.5]"
  "soft_exp0p25_mc0p75 [0.25,0.75]"
  "soft_exp0p10_mc0p90 [0.1,0.9]"
)

for entry in "${SOFT_CELLS[@]}"; do
    set -- $entry
    name="$1"
    weights="$2"
    if [[ -n "$ONLY_CELL" && "$name" != "$ONLY_CELL" ]]; then continue; fi
    run_cell "$name" soft "$weights"
done

for thr in 0.9 0.75 0.5; do
    safe_thr="$(printf '%s' "$thr" | tr . p)"
    name="hard_t${safe_thr}"
    if [[ -n "$ONLY_CELL" && "$name" != "$ONLY_CELL" ]]; then continue; fi
    run_cell "$name" hard "[0.5,0.5]" "$thr"
done

log "All cells finished. Summary will be aggregated by summary.py"
