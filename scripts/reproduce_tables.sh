#!/usr/bin/env bash
# One-shot driver: compute scalar + ordering + modality-stratified metrics
# for every (model, prompt) cell with cached predictions.

set -euo pipefail

OUT_ROOT="${MMSCALE_OUTPUT_ROOT:-./outputs}"
PRED_DIR="${OUT_ROOT}/predictions"
METRICS_DIR="${OUT_ROOT}/metrics"
mkdir -p "$METRICS_DIR"

MODELS=(llava_onevision_7b qwen2_vl_7b qwen2_5_vl_7b qwen3_vl_8b
        gpt5_2 claude_opus_4_7 gemini_2_5_pro)
PROMPTS=(zero_shot icl3 cot)

for m in "${MODELS[@]}"; do
  for p in "${PROMPTS[@]}"; do
    pred="${PRED_DIR}/${m}_${p}.jsonl"
    [[ -f "$pred" ]] || { echo "skip $m/$p (missing predictions)"; continue; }
    python -m src.evaluate_scalar              --predictions "$pred" --output "${METRICS_DIR}/${m}_${p}_scalar.json"
    python -m src.evaluate_ordering            --predictions "$pred" --output "${METRICS_DIR}/${m}_${p}_ordering.json"
    python -m src.evaluate_modality_stratified --predictions "$pred" --output "${METRICS_DIR}/${m}_${p}_modality.json"
  done
done

echo
echo "[done]  metrics: $METRICS_DIR"
