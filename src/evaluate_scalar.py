"""Scalar-rating metrics: MAE, RMSE, AUC-acceptability, ECE.

These are sample-level metrics over all scenarios with a parsed prediction.
Use src/evaluate_ordering.py for within-image (group-level) metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.utils import read_jsonl, write_json


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    err = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(err ** 2)))


def auc_acceptability(y_true, y_pred, threshold: float = 2.5) -> float:
    labels = (np.asarray(y_true) >= threshold).astype(int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(y_pred)))


def ece(y_true, y_pred, n_bins: int = 10, threshold: float = 2.5) -> float:
    pred = np.asarray(y_pred, dtype=float)
    true = (np.asarray(y_true, dtype=float) >= threshold).astype(float)
    conf = np.clip((pred - 1.0) / 4.0, 0.0, 1.0)
    bins = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for i in range(n_bins):
        mask = (conf >= bins[i]) & (conf < bins[i + 1])
        if not np.any(mask):
            continue
        acc = true[mask].mean()
        avg_conf = conf[mask].mean()
        out += mask.mean() * abs(acc - avg_conf)
    return float(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    rows = [r for r in read_jsonl(args.predictions) if r.get("prediction") is not None]
    y_true = [r["gold_rating"] for r in rows]
    y_pred = [r["prediction"]  for r in rows]

    metrics = {
        "n_predictions": len(rows),
        "n_total":       sum(1 for _ in open(args.predictions)),
        "mae":           mae(y_true, y_pred),
        "rmse":          rmse(y_true, y_pred),
        "auc_acceptability_2.5": auc_acceptability(y_true, y_pred, 2.5),
        "ece_2.5":       ece(y_true, y_pred, threshold=2.5),
    }
    print(json.dumps(metrics, indent=2))
    if args.output:
        write_json(metrics, args.output)


if __name__ == "__main__":
    main()
