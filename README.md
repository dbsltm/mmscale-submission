# MM-SCALE: Multimodal Scenario Calibration And Legibility Evaluation

Code release for the MM-SCALE benchmark and the rationale-monitorability evaluation methodology proposed in our NeurIPS 2026 E&B Track submission. The benchmark and methodology probe whether vision-language models (VLMs) produce **monitorable rationales** when making visually grounded moral judgments, decomposed into three levels: legibility, human-grounding alignment, and behavioral support.

## Layout

```
mmscale-submission/
├── README.md                    this file
├── LICENSE
├── requirements.txt
├── dataset_card.md              human-readable dataset description (data is hosted on Hugging Face)
├── configs/                     YAML configs for paths, models, prompts
├── prompts/                     verbatim model & verifier prompts
├── src/                         core Python modules (data / inference / metrics)
└── scripts/                     bash drivers for each task
```

The dataset itself (images, labels, splits) is **not** in this repo — it lives on Hugging Face. See `dataset_card.md` for the URL and the expected directory layout under `MMSCALE_DATA_ROOT`.

## Quick start

```bash
pip install -r requirements.txt

# 1. Download the dataset from Hugging Face (URL in dataset_card.md)
#    and point MMSCALE_DATA_ROOT at the unpacked directory.
export MMSCALE_DATA_ROOT=/path/to/mmscale

# 2. Run the headline scalar-rating eval on a small open-weight model.
bash scripts/run_task1_scalar.sh qwen2_5_vl_7b zero_shot

# 3. Compute within-image ordering metrics (NDCG@5 + pairwise accuracy at Δ≥1).
bash scripts/run_task2_ordering.sh qwen2_5_vl_7b zero_shot

# 4. Run the CoT-modality verifier on a CoT predictions file.
bash scripts/run_task3_verifier.sh outputs/predictions/qwen2_5_vl_7b_cot.jsonl

# 5. One-shot: run all metrics for every (model, prompt) with cached predictions.
bash scripts/reproduce_tables.sh
```

## API keys

Closed-source inference reads keys from environment variables only. Set whichever you need:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
```

No keys, paths, or author identifiers are committed to this repo.

## Headline metrics

- **Aggregate:** NDCG@5 (per-image, then averaged).
- **Within-image:** pairwise accuracy at $\Delta\bar{y}\geq 0$ and $\Delta\bar{y}\geq 1$, where $\Delta\bar{y}$ is the absolute difference between human mean ratings within a pair. Tied predictions count $0.5$ credit.
- **Modality-stratified pairwise accuracy:** same metric, partitioned by the modality combination of the pair (T-T / I-I / B-B / T-I / T-B / I-B).
- **Rationale grounding (verifier):** five-way load-bearing label (`text-grounded` / `image-grounded` / `both` / `ungrounded` / `hallucinated-visual`) over CoT traces. Verifier prompt at `prompts/verifier_evidence_grounding.txt`.

## What's included vs. left out

This release contains the headline pipeline only — enough to reproduce the main paper tables and figures. Two parts of the full research codebase are deliberately not included here:

1. The full sweep of training experiments (LoRA / DPO / listwise) used in §5.2 of the paper.
2. The complete data-generation pipeline (DALL·E image generation, scenario authoring, rating collection). The released dataset is the cleaned output; the construction pipeline is described in `dataset_card.md` and §3 of the paper.

Reach out via the issue tracker after de-anonymization for access.

## Reproducibility

All entry-point scripts read seeds from `configs/paths.yaml` (default 42).

## Citation

```bibtex
@inproceedings{anonymous2026mmscale,
  title     = {MM-SCALE: A Methodology for Evaluating Rationale Monitorability
               in Visually Grounded Moral Judgment},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems
               (Datasets and Benchmarks Track)},
  year      = {2026}
}
```
