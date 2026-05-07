# MM-SCALE: Evaluating Evidence-Grounded Moral Judgment in Vision-Language Models

Anonymous code release for the MM-SCALE benchmark and evaluation pipeline proposed in the accompanying NeurIPS 2026 Datasets & Benchmarks Track submission.

MM-SCALE is an image-centered benchmark for evaluating **evidence-grounded moral judgment** in vision-language models (VLMs). Each image context is paired with multiple action scenarios, allowing evaluation beyond verdict-level agreement: models must assign scalar moral acceptability scores, preserve human moral orderings among scenarios sharing the same image, and ground rationales in the evidence source humans judged to be critical.

The benchmark contains:

- 8,444 image contexts
- 21,977 action scenarios
- human mean moral acceptability ratings on a 1-5 scale
- scenario-level modality-grounding labels: `text`, `image`, or `both`

The main evaluation question is not only whether a model reaches a plausible moral verdict, but whether that verdict follows from the appropriate evidence source.

## Repository layout

```text
mmscale-submission/
├── README.md
├── LICENSE
├── requirements.txt
├── dataset_card.md
├── .gitignore
├── configs/
│   ├── paths.yaml
│   ├── models.yaml
│   └── prompts.yaml
├── prompts/
│   └── ...
├── src/
│   ├── data.py
│   ├── utils.py
│   ├── parse_outputs.py
│   ├── run_model.py
│   ├── evaluate_scalar.py
│   ├── evaluate_ordering.py
│   ├── evaluate_modality_stratified.py
│   └── verify_rationales.py
└── scripts/
    ├── run_task1_scalar.sh
    ├── run_task2_ordering.sh
    ├── run_task3_verifier.sh
    └── reproduce_tables.sh