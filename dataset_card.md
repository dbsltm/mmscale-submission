# MM-SCALE Dataset Card

## At a glance

- **Task:** within-image moral acceptability ordering and modality attribution
- **Size:** 8,444 image contexts / 21,977 scenarios (1–5 morality scale)
- **Images:** synthetic (DALL·E generated)
- **Splits:** image-disjoint train/val/test (5,910 / 844 / 1,690 images)
- **Modality labels:** `text` (18.3%) / `image` (46.9%) / `both` (34.8%)

## Where to get the data

The dataset (images + JSONL labels + per-task split files) is hosted on Hugging Face:

> **`<HF_DATASET_URL>`**  *(URL provided in the data-availability statement of the paper)*

After downloading, set:

```bash
export MMSCALE_DATA_ROOT=/path/to/mmscale
```

Files expected under `MMSCALE_DATA_ROOT`:

```
images/                          *.png, one per image_id
mmscale_clean_flat.jsonl         21,977 rows (one per scenario)
mmscale_clean_contexts.jsonl     8,444 rows (one per image, scenarios nested)
splits/
    train.jsonl  val.jsonl  test.jsonl                    pointwise
    train_binary.jsonl  val_binary.jsonl  test_binary.jsonl
    train_pairwise.jsonl  val_pairwise.jsonl  test_pairwise.jsonl
    train_listwise.jsonl  val_listwise.jsonl  test_listwise.jsonl
```

## Schema (`mmscale_clean_flat.jsonl` — one row per scenario)

| field | type | description |
|---|---|---|
| `image_id` | string | filename of the source image, e.g. `03485.png` |
| `target_setting` | string | the moral situation depicted by the image |
| `scenario_id` | string | unique id, format `<image_id>_s<NN>` |
| `text` | string | the textual scenario being judged |
| `mean_rating` | float | mean 1–5 morality rating across annotators |
| `ratings` | list[int] | individual annotator ratings (1–5) |
| `rating_std` | float | standard deviation of ratings |
| `modality` | string | aggregated label: `text` / `image` / `both` |
| `modality_votes` | list[string] | per-annotator modality votes |
| `source_slots` | list[int] | which slot(s) in the source survey this item came from |
| `dedupe_group_size` | int | how many near-duplicates were collapsed into this item |

`mmscale_clean_contexts.jsonl` carries the same fields, but `scenarios` is a nested list of per-scenario records and the row also exposes `image_url`.

## Construction

Image prompts and scenario texts were derived from a moral-situation collection task. We then generated images with **DALL·E** conditioned on each `target_setting`, and collected scenario judgments via crowdworkers who rated each (image, scenario) pair on a 1–5 morality scale and selected the modality (`image` / `text` / `both`) that drove their judgment.

Cleaning pipeline (full report in `cleaning_report.json` in the data release):
- Drop scenarios with conflicting ratings (spread > 1.0): −2,065
- Drop scenarios with missing text: −14,057
- Drop images with fewer than 2 surviving scenarios: −802 images
- Near-duplicate collapsing at cosine similarity ≥ 0.88: 6,937 groups merged

## Responsible AI

- **No personally identifiable information.** Images are synthetic; scenario texts describe generic situations without real names, locations, or events.
- **Some scenarios describe morally negative or distressing situations** (interpersonal harm, deception, substance use). This is required for moral-judgment evaluation. The dataset deliberately excludes CSAM, instructions for real-world harm, graphic violence, and self-harm encouragement.
- **Known biases:** synthetic-image distribution shift; English-only; aggregated single rating per item flattens legitimate moral disagreement (per-annotator ratings are retained for pluralism studies); DALL·E artifacts may correlate with modality labels in uncharacterized ways.
- **Intended use:** benchmarking VLM rationale monitorability, multimodal grounding, modality attribution, methodological studies of rationale faithfulness.
- **Out-of-scope:** training production content-moderation systems without further validation; inferring moral judgments about identifiable real individuals or events.

The full Croissant Responsible AI metadata block is shipped alongside the dataset on Hugging Face as `croissant_rai.json`.

## Maintenance

Versioned on the data host. Errata and label corrections tracked in the host platform's `CHANGELOG.md`. Contact via the issue tracker on this repo (post-anonymization).

## License

CC BY 4.0 for the compilation, scenario texts, and annotations. DALL·E images are additionally subject to OpenAI's content policy at the time of generation.
