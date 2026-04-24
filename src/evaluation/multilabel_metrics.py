"""
Multi-label Metrics for Problem B Category Classification
===========================================================
Per-label F1, macro/micro F1, Hamming loss, subset accuracy.
"""

import numpy as np
from sklearn.metrics import (
    f1_score, hamming_loss, accuracy_score,
    classification_report, precision_recall_fscore_support,
)
from typing import Dict, Any, List


def multilabel_report(y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str]) -> Dict[str, Any]:
    """Full multi-label evaluation report."""
    per_label_p, per_label_r, per_label_f1, per_label_support = (
        precision_recall_fscore_support(y_true, y_pred, average=None)
    )
    per_label = {}
    for i, name in enumerate(label_names):
        per_label[name] = {
            "precision": float(per_label_p[i]),
            "recall": float(per_label_r[i]),
            "f1": float(per_label_f1[i]),
            "support": int(per_label_support[i]),
        }
    return {
        "per_label": per_label,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
    }
