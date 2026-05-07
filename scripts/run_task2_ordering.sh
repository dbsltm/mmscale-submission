#!/usr/bin/env bash
# Task 2: within-image ordering — NDCG@5 + pairwise accuracy at Δ=0 and Δ≥1.
# Assumes scalar predictions already exist from Task 1.
#
# Usage:  bash scripts/run_task2_ordering.sh <model_key> <prompt_type>

set -euo pipefail
MODEL="${1:?model key required}"
PROMPT="${2:?prompt type required}"

OUT_ROOT="${MMSCALE_OUTPUT_ROOT:-./outputs}"
PRED="${OUT_ROOT}/predictions/${MODEL}_${PROMPT}.jsonl"
ORDERING="${OUT_ROOT}/metrics/${MODEL}_${PROMPT}_ordering.json"

if [[ ! -f "$PRED" ]]; then
  echo "missing predictions: $PRED"
  echo "run scripts/run_task1_scalar.sh $MODEL $PROMPT first"
  exit 1
fi

python -m src.evaluate_ordering --predictions "$PRED" --output "$ORDERING"

echo
echo "[done]  ordering metrics: $ORDERING"
