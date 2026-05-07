#!/usr/bin/env bash
# Task 3: CoT-modality verifier.
# Usage:  bash scripts/run_task3_verifier.sh <cot_predictions.jsonl> [judge_provider] [judge_model]

set -euo pipefail
PRED="${1:?cot predictions JSONL required}"
JUDGE_PROVIDER="${2:-openai}"
JUDGE_MODEL="${3:-gpt-4o-2024-08-06}"

OUT_ROOT="${MMSCALE_OUTPUT_ROOT:-./outputs}"
BASE="$(basename "$PRED" .jsonl)"
OUT="${OUT_ROOT}/verifier/${BASE}_judged.jsonl"

mkdir -p "$(dirname "$OUT")"

python -m src.verify_rationales \
  --predictions "$PRED" \
  --output "$OUT" \
  --judge_provider "$JUDGE_PROVIDER" \
  --judge_model "$JUDGE_MODEL"

echo
echo "[done]  verifier output: $OUT"
