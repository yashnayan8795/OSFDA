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


def class_weighted_ordinal_mae(y_true, y_pred) -> float:
    """
    Ordinal MAE weighted by inverse class frequency.

    Rare classes (e.g., Level-3 critical incidents) receive higher weight,
    preventing majority-class dominance of the plain MAE metric.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes, counts = np.unique(y_true, return_counts=True)
    inv_freq = 1.0 / counts.astype(float)
    inv_freq /= inv_freq.sum()
    weight_map = dict(zip(classes, inv_freq))
    sample_weights = np.array([weight_map[c] for c in y_true])
    abs_errors = np.abs(y_true - y_pred).astype(float)
    return float(np.average(abs_errors, weights=sample_weights))


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
    import logging
    logger = logging.getLogger(__name__)

    rng = np.random.RandomState(seed)
    scores = []
    n = len(y_true)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_pred[idx]))

    n_actual = len(scores)
    n_skipped = n_boot - n_actual
    if n_skipped > n_boot * 0.1:
        logger.warning(
            "bootstrap_metric: %d/%d samples skipped (%.1f%%) due to "
            "insufficient class diversity. CI may be unreliable.",
            n_skipped, n_boot, 100 * n_skipped / n_boot,
        )

    lo = np.percentile(scores, (100 - ci) / 2)
    hi = np.percentile(scores, 100 - (100 - ci) / 2)
    return {"mean": np.mean(scores), "ci_low": lo, "ci_high": hi,
            "n_boot_requested": n_boot, "n_boot_actual": n_actual}


def full_severity_report(y_true, y_pred, cost_matrix: Optional[dict] = None) -> Dict[str, Any]:
    """Comprehensive evaluation report for severity predictions."""
    report = {
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "ordinal_mae": ordinal_mae(y_true, y_pred),
        "class_weighted_mae": class_weighted_ordinal_mae(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
        "qwk_bootstrap": bootstrap_metric(y_true, y_pred, quadratic_weighted_kappa),
    }
    if cost_matrix:
        report["asymmetric_cost"] = asymmetric_cost(y_true, y_pred, cost_matrix)
    return report
