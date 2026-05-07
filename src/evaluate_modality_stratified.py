"""Modality-stratified within-image pairwise accuracy.

For each image, partition all qualifying scenario pairs by the modality
combination of the two scenarios (T-T / I-I / B-B / T-I / T-B / I-B) and
report pairwise accuracy in each bucket. Reproduces Figure 2 of the paper.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from src.utils import read_jsonl, write_json


PAIR_TYPES = ["T-T", "I-I", "B-B", "T-I", "T-B", "I-B"]
MOD_LETTER = {"text": "T", "image": "I", "both": "B"}


def pair_type(m1: str, m2: str) -> str:
    a, b = MOD_LETTER[m1], MOD_LETTER[m2]
    if a == b:
        return f"{a}-{a}"
    pretty = {"BI": "I-B", "BT": "T-B", "IT": "T-I"}
    return pretty["".join(sorted([a, b]))]


def stratified(rows: list[dict], delta: float) -> dict:
    """Returns {pair_type: {acc, n, ci_lo, ci_hi (Wilson 95%)}}."""
    groups = defaultdict(list)
    for r in rows:
        if r.get("prediction") is None:
            continue
        if r.get("modality") not in MOD_LETTER:
            continue
        groups[r["image_id"]].append(r)

    correct_x2: dict[str, int] = defaultdict(int)
    n: dict[str, int] = defaultdict(int)
    for items in groups.values():
        if len(items) < 2:
            continue
        for a, b in combinations(items, 2):
            ga, gb = float(a["gold_rating"]), float(b["gold_rating"])
            pa, pb = float(a["prediction"]),  float(b["prediction"])
            d = abs(ga - gb)
            if d < delta or d == 0:
                continue
            pt = pair_type(a["modality"], b["modality"])
            n[pt] += 1
            if pa == pb:
                correct_x2[pt] += 1
            elif (pa > pb) == (ga > gb):
                correct_x2[pt] += 2

    out = {}
    for pt in PAIR_TYPES:
        if n[pt] == 0:
            out[pt] = {"acc": float("nan"), "n": 0,
                       "ci_lo": float("nan"), "ci_hi": float("nan")}
            continue
        acc = (correct_x2[pt] / 2) / n[pt]
        # Wilson 95% interval
        z, p, nn = 1.96, acc, n[pt]
        denom = 1 + z**2 / nn
        center = (p + z**2 / (2 * nn)) / denom
        rad = (z / denom) * np.sqrt(p * (1 - p) / nn + z**2 / (4 * nn**2))
        out[pt] = {"acc": acc, "n": nn,
                   "ci_lo": center - rad, "ci_hi": center + rad}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--delta", type=float, default=0.0)
    args = ap.parse_args()

    rows = read_jsonl(args.predictions)
    by_pt = stratified(rows, args.delta)
    out = {"delta": args.delta, "by_pair_type": by_pt}
    print(json.dumps(out, indent=2))
    if args.output:
        write_json(out, args.output)


if __name__ == "__main__":
    main()
