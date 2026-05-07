"""Dataset loaders and split helpers.

The dataset is hosted externally (see dataset_card.md). Set MMSCALE_DATA_ROOT
to the directory you downloaded it into; the loaders read from paths.yaml,
which expands ${MMSCALE_DATA_ROOT}.

Expected files under MMSCALE_DATA_ROOT:
    images/                              one PNG per image_id
    mmscale_clean_flat.jsonl             one row per scenario
    mmscale_clean_contexts.jsonl         one row per image (scenarios nested)
    splits/
        train.jsonl  val.jsonl  test.jsonl   image-disjoint
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from src.utils import load_yaml, read_jsonl


def load_paths(cfg_path: str = "configs/paths.yaml") -> dict:
    return load_yaml(cfg_path)


def _split_scenario_ids(paths: dict, split: str) -> set[str]:
    split_path = Path(paths["splits_dir"]) / f"{split}.jsonl"
    keep: set[str] = set()
    with split_path.open() as f:
        for line in f:
            ctx = json.loads(line)
            for s in ctx.get("scenarios", []):
                keep.add(s["scenario_id"])
    return keep


def load_flat(paths: dict | None = None,
              split: str | None = None) -> list[dict]:
    """Load the flat (one row per scenario) records, optionally filtered to a split."""
    paths = paths or load_paths()
    rows = read_jsonl(paths["flat_jsonl"])
    if split is None:
        return rows
    keep = _split_scenario_ids(paths, split)
    return [r for r in rows if r["scenario_id"] in keep]


def load_contexts(paths: dict | None = None,
                  split: str | None = None) -> list[dict]:
    """Load the image-grouped records (one row per image, scenarios nested)."""
    paths = paths or load_paths()
    rows = read_jsonl(paths["contexts_jsonl"])
    if split is None:
        return rows
    keep = _split_scenario_ids(paths, split)
    out = []
    for r in rows:
        scens = [s for s in r["scenarios"] if s["scenario_id"] in keep]
        if scens:
            out.append({**r, "scenarios": scens})
    return out


def resolve_image_path(image_id: str, paths: dict | None = None) -> Path:
    paths = paths or load_paths()
    return Path(paths["images_dir"]) / image_id


def sample_icl_exemplars(pool_path: str, k: int = 3, seed: int = 42) -> list[dict]:
    """Sample k exemplars with a fixed seed. Pool is a JSONL of {text, mean_rating}."""
    pool = read_jsonl(pool_path)
    rng = random.Random(seed)
    return rng.sample(pool, k)
