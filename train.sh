#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# train.sh  –  Retrain PCPL-PPO on all GraphAllocBench problems.
#
# Reads best hyperparameters from the sweep CSV and trains 5 seeds × 1M steps
# for every problem in the main benchmark (problems 0–5).
#
# Usage:
#   ./train.sh                              # all problems, seeds 0-4, 1M steps
#   ./train.sh --seeds 0 1 2               # specific seeds
#   ./train.sh --problems problem_0 problem_1a
#   ./train.sh --total-steps 500000
#   ./train.sh --workers 4                 # cap parallel processes
#   ./train.sh --wandb-mode disabled       # suppress W&B (pure local run)
#   ./train.sh --evaluate                  # run evaluation after training
#
#   ./train.sh --gnn                       # also train GNN problems (6a/6b/6c, archs 0/2/3, 3M steps)
#   ./train.sh --gnn --no-main             # GNN only, skip main benchmark
#   ./train.sh --gnn --gnn-problems problem_6a problem_6b
#   ./train.sh --gnn --gnn-archs 0 2       # specific GNN architectures
#
# --evaluate runs evaluate.py after all training finishes and writes:
#   graphallocbench/examples/data/pcpl_stats.csv         (main benchmark)
#   graphallocbench/examples/data/GraphAllocBench-GNN-v1-1/pcpl_stats_gnn.csv  (GNN)
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Ctrl-C / SIGTERM: kill the entire process group immediately ──────────────
trap 'echo ""; echo "Interrupted — killing all training processes..."; kill 0; exit 1' INT TERM

# ── defaults ────────────────────────────────────────────────────────────────
TOTAL_STEPS=1000000
SEEDS=(0 1 2 3 4)
WANDB_MODE="${WANDB_MODE:-online}"   # override with: WANDB_MODE=disabled ./train.sh
EVALUATE=false

# ── Python check ────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found on PATH" >&2
    exit 1
fi

# Prefer the venv if it exists next to this script
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# ── sanity check: CSV must exist ────────────────────────────────────────────
CSV="$SCRIPT_DIR/graphallocbench/examples/data/GraphAllocBench-v2/best_hyperparameters.csv"
if [ ! -f "$CSV" ]; then
    echo "ERROR: best_hyperparameters.csv not found at:" >&2
    echo "  $CSV" >&2
    echo "Run the W&B sweep first, then export best configs to that path." >&2
    exit 1
fi

# ── intercept --evaluate; pass everything else through ──────────────────────
PASS_THROUGH=()
for arg in "$@"; do
    if [ "$arg" = "--evaluate" ]; then
        EVALUATE=true
    else
        PASS_THROUGH+=("$arg")
    fi
done

# ── training ─────────────────────────────────────────────────────────────────
echo "=== GraphAllocBench PCPL-PPO Training ==="
echo "  Repo    : $SCRIPT_DIR"
echo "  Steps   : $TOTAL_STEPS"
echo "  W&B     : $WANDB_MODE"
echo "  Evaluate: $EVALUATE"
echo ""

WANDB_MODE="$WANDB_MODE" python3 train.py \
    --csv         "$CSV" \
    --total-steps "$TOTAL_STEPS" \
    --seeds       "${SEEDS[@]}" \
    --skip-done \
    ${PASS_THROUGH[@]+"${PASS_THROUGH[@]}"}

# ── evaluation (optional) ─────────────────────────────────────────────────
if $EVALUATE; then
    echo ""
    echo "=== GraphAllocBench PCPL-PPO Evaluation ==="
    python3 evaluate.py \
        --csv         "$CSV" \
        --total-steps "$TOTAL_STEPS" \
        --seeds       "${SEEDS[@]}" \
        ${PASS_THROUGH[@]+"${PASS_THROUGH[@]}"}
    echo ""
    echo "pcpl_stats.csv written to graphallocbench/examples/data/"
fi
