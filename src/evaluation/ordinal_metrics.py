"""
Ordinal Metrics for Problem A Severity
========================================
QWK, ordinal MAE, per-class metrics with bootstrap CIs.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
from typing import Dict, Any, Optional


def quadratic_weighted_kappa(y_true, y_pred) -> float:
    """Quadratic Weighted Kappa — primary metric for ordinal severity."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def ordinal_mae(y_true, y_pred) -> float:
    """Mean Absolute Error on ordinal levels."""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def asymmetric_cost(y_true, y_pred, cost_matrix: dict) -> float:
    """Compute average asymmetric cost from the cost matrix config."""
    total = 0.0
    for t, p in zip(y_true, y_pred):
        row_key = f"actual_{t}"
        col_key = f"pred_{p}"
        total += cost_matrix.get(row_key, {}).get(col_key, 0)
    return total / len(y_true)


def bootstrap_metric(y_true, y_pred, metric_fn, n_boot=1000, ci=95, seed=42):
    """Bootstrap a metric to get confidence interval."""
    rng = np.random.RandomState(seed)
    scores = []
    n = len(y_true)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    lo = np.percentile(scores, (100 - ci) / 2)
    hi = np.percentile(scores, 100 - (100 - ci) / 2)
    return {"mean": np.mean(scores), "ci_low": lo, "ci_high": hi, "n_boot": len(scores)}


def full_severity_report(y_true, y_pred, cost_matrix: Optional[dict] = None) -> Dict[str, Any]:
    """Comprehensive evaluation report for severity predictions."""
    report = {
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "ordinal_mae": ordinal_mae(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
        "qwk_bootstrap": bootstrap_metric(y_true, y_pred, quadratic_weighted_kappa),
    }
    if cost_matrix:
        report["asymmetric_cost"] = asymmetric_cost(y_true, y_pred, cost_matrix)
    return report
