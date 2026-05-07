"""Within-image ordering metrics: NDCG@5 and within-image pairwise accuracy
at multiple Δ thresholds.

Pairwise accuracy is the headline metric in the paper. Tied predictions count
0.5 credit; pairs with |Δ gold| < delta or = 0 are excluded.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations

import numpy as np

from src.utils import read_jsonl, write_json


# ---------------------- aggregate ranking metrics ----------------------

def _dcg(scores: np.ndarray, k: int | None = None) -> float:
    if k is not None:
        scores = scores[:k]
    return float(np.sum((2 ** scores - 1) / np.log2(np.arange(len(scores)) + 2)))


def ndcg_at_k(y_true, y_pred, k: int = 5) -> float:
    if len(y_true) == 0:
        return 0.0
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    order = np.argsort(y_pred)[::-1]
    ideal = np.argsort(y_true)[::-1]
    denom = _dcg(y_true[ideal], k=k)
    if denom <= 0:
        return 0.0
    return _dcg(y_true[order], k=k) / denom


def per_image_ndcg(rows: list[dict]) -> tuple[list[str], np.ndarray]:
    """Returns (image_ids, ndcg_per_image) for groups with >=2 valid items."""
    groups = defaultdict(list)
    for r in rows:
        if r.get("prediction") is not None:
            groups[r["image_id"]].append(r)
    ids, ndcgs = [], []
    for img, items in groups.items():
        if len(items) < 2:
            continue
        y_true = [float(x["gold_rating"]) for x in items]
        y_pred = [float(x["prediction"])  for x in items]
        ids.append(img)
        ndcgs.append(ndcg_at_k(y_true, y_pred, k=5))
    return ids, np.asarray(ndcgs)


# ---------------------- pairwise-accuracy (headline) ----------------------

def pairs_per_image(rows: list[dict]) -> dict[str, list[tuple[float, float, float, float]]]:
    """{image_id: [(gold_a, gold_b, pred_a, pred_b), ...]}."""
    groups = defaultdict(list)
    for r in rows:
        if r.get("prediction") is not None:
            groups[r["image_id"]].append(r)
    out = {}
    for img, items in groups.items():
        if len(items) < 2:
            continue
        scenes = [(float(x["gold_rating"]), float(x["prediction"])) for x in items]
        out[img] = [(ga, gb, pa, pb)
                    for (ga, pa), (gb, pb) in combinations(scenes, 2)]
    return out


def pairwise_accuracy(pairs_by_img: dict, delta: float) -> tuple[float, int]:
    """Pooled accuracy over qualifying pairs (|Δ gold| >= delta and != 0).
    Tied predictions = 0.5 credit. Returns (accuracy, n_qualifying_pairs)."""
    n_qual = correct_x2 = 0
    for ps in pairs_by_img.values():
        for ga, gb, pa, pb in ps:
            d = abs(ga - gb)
            if d < delta or d == 0:
                continue
            n_qual += 1
            if pa == pb:
                correct_x2 += 1
            elif (pa > pb) == (ga > gb):
                correct_x2 += 2
    return ((correct_x2 / 2) / n_qual if n_qual else float("nan"),
            n_qual)


# ---------------------- main ----------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.0, 1.0])
    args = ap.parse_args()

    rows = read_jsonl(args.predictions)
    ids, ndcgs = per_image_ndcg(rows)
    pairs = pairs_per_image(rows)

    metrics = {
        "n_predictions": sum(1 for r in rows if r.get("prediction") is not None),
        "n_image_groups": len(ids),
        "ndcg_at_5_mean": float(ndcgs.mean()) if len(ndcgs) else 0.0,
        "pairwise_accuracy": {},
    }
    for d in args.deltas:
        acc, n = pairwise_accuracy(pairs, d)
        metrics["pairwise_accuracy"][f"delta>={d:g}"] = {"acc": acc, "n_pairs": n}

    print(json.dumps(metrics, indent=2))
    if args.output:
        write_json(metrics, args.output)


if __name__ == "__main__":
    main()
