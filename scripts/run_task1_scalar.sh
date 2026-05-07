#!/usr/bin/env bash
# Task 1: scalar moral-acceptability rating.
# Usage:  bash scripts/run_task1_scalar.sh <model_key> <prompt_type> [split]
#         model_key  ∈  configs/models.yaml/models
#         prompt_type ∈ zero_shot | icl3 | cot
#         split      ∈ test | val | train  (default: test)

set -euo pipefail
MODEL="${1:?model key required}"
PROMPT="${2:?prompt type required}"
SPLIT="${3:-test}"

OUT_ROOT="${MMSCALE_OUTPUT_ROOT:-./outputs}"
PRED="${OUT_ROOT}/predictions/${MODEL}_${PROMPT}.jsonl"
METRICS="${OUT_ROOT}/metrics/${MODEL}_${PROMPT}.json"

mkdir -p "$(dirname "$PRED")" "$(dirname "$METRICS")"

python -m src.run_model \
  --model "$MODEL" \
  --prompt "$PROMPT" \
  --split "$SPLIT" \
  --output "$PRED"

python -m src.evaluate_scalar \
  --predictions "$PRED" \
  --output "$METRICS"

echo
echo "[done]  predictions: $PRED"
echo "        metrics:     $METRICS"
