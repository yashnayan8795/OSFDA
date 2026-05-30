"""
Experiment Logger — Multi-Metric Comparison Table
===================================================
Accumulates results across Problem A experiments and prints a
formatted comparison table. Ensures every experiment reports
the full required metric set before it is accepted.

Usage
-----
    from src.evaluation.experiment_logger import ExperimentLogger

    log = ExperimentLogger()
    log.add_result("CatBoost PCA-32", qwk=0.563, macro_f1=0.349, ...)
    log.add_result("CatBoost PCA-128", qwk=0.591, ...)
    log.print_comparison_table()

    # Top misclassifications review (required before accepting a model)
    log.top_n_misclassifications(X_test, y_test, y_pred, probs, costs, n=100)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Required metrics for any model to be accepted
# ---------------------------------------------------------------------------
REQUIRED_METRICS = [
    "qwk",
    "macro_f1",
    "l3_recall",
    "l3_precision",
    "expected_cost",
    "ece_mean",
]

# Acceptance thresholds relative to baseline (delta gates)
ACCEPTANCE_GATES = {
    "qwk_drop_max": 0.02,          # QWK must not drop by more than 0.02 vs baseline
    "ece_max": 0.035,               # ECE must not exceed 0.035 after calibration
    "l3_precision_min": 0.05,       # Level-3 Precision must be at least 5%
}


class ExperimentLogger:
    """Accumulates per-experiment metrics and generates comparison tables."""

    def __init__(self, baseline_name: Optional[str] = None):
        self.results: List[Dict] = []
        self.baseline_name = baseline_name
        self._baseline_qwk: Optional[float] = None

    def add_result(
        self,
        name: str,
        qwk: float,
        macro_f1: float,
        l3_recall: float,
        l3_precision: float,
        expected_cost: float,
        ece_mean: float,
        confusion_matrix: Optional[List] = None,
        notes: str = "",
        is_baseline: bool = False,
    ) -> Dict:
        """
        Record a single experiment result.

        Returns
        -------
        dict with 'accepted' key indicating if the model passes the gates.
        """
        row = {
            "name": name,
            "qwk": qwk,
            "macro_f1": macro_f1,
            "l3_recall": l3_recall,
            "l3_precision": l3_precision,
            "expected_cost": expected_cost,
            "ece_mean": ece_mean,
            "confusion_matrix": confusion_matrix,
            "notes": notes,
            "is_baseline": is_baseline,
            "accepted": True,
            "rejection_reasons": [],
        }

        if is_baseline or (self.baseline_name and name == self.baseline_name):
            self._baseline_qwk = qwk

        # Apply acceptance gates if baseline is known
        if self._baseline_qwk is not None and not (is_baseline or name == self.baseline_name):
            reasons = []
            if (self._baseline_qwk - qwk) > ACCEPTANCE_GATES["qwk_drop_max"]:
                reasons.append(
                    f"QWK dropped by {self._baseline_qwk - qwk:.3f} "
                    f"(max allowed: {ACCEPTANCE_GATES['qwk_drop_max']})"
                )
            if ece_mean > ACCEPTANCE_GATES["ece_max"]:
                reasons.append(
                    f"ECE {ece_mean:.4f} > threshold {ACCEPTANCE_GATES['ece_max']}"
                )
            if l3_precision < ACCEPTANCE_GATES["l3_precision_min"]:
                reasons.append(
                    f"Level-3 Precision {l3_precision:.3f} < "
                    f"minimum {ACCEPTANCE_GATES['l3_precision_min']}"
                )
            if reasons:
                row["accepted"] = False
                row["rejection_reasons"] = reasons

        self.results.append(row)
        return row

    def print_comparison_table(self) -> None:
        """Print a formatted Markdown comparison table."""
        if not self.results:
            print("No experiments recorded.")
            return

        header = (
            f"{'Model':<35} | {'QWK':>6} | {'F1':>6} | "
            f"{'L3-R':>6} | {'L3-P':>6} | {'Cost':>7} | {'ECE':>6} | {'Status'}"
        )
        sep = "-" * len(header)
        print("\n" + "=" * len(header))
        print("  PROBLEM A — MODEL COMPARISON TABLE")
        print("=" * len(header))
        print(header)
        print(sep)

        for r in self.results:
            status = "✓ BASELINE" if r["is_baseline"] else ("✓ ACCEPTED" if r["accepted"] else "✗ REJECTED")
            print(
                f"{r['name']:<35} | {r['qwk']:>6.4f} | {r['macro_f1']:>6.4f} | "
                f"{r['l3_recall']:>6.3f} | {r['l3_precision']:>6.3f} | "
                f"{r['expected_cost']:>7.4f} | {r['ece_mean']:>6.4f} | {status}"
            )
            for reason in r.get("rejection_reasons", []):
                print(f"  {'':35}   REJECTED: {reason}")

        print("=" * len(header))

        # Print best accepted
        accepted = [r for r in self.results if r["accepted"]]
        if accepted:
            best = max(accepted, key=lambda r: r["qwk"])
            print(f"\n  Best accepted model: {best['name']} (QWK={best['qwk']:.4f})")

    def print_confusion_matrix(self, name: str) -> None:
        """Print confusion matrix for a specific experiment."""
        for r in self.results:
            if r["name"] == name and r["confusion_matrix"] is not None:
                cm = r["confusion_matrix"]
                print(f"\nConfusion Matrix — {name}:")
                print(f"  {'':>10} pred_0  pred_1  pred_2  pred_3")
                for i, row in enumerate(cm):
                    print(f"  actual_{i}  {row[0]:>6}  {row[1]:>6}  {row[2]:>6}  {row[3]:>6}")
                return
        print(f"No confusion matrix found for: {name}")


def top_n_misclassifications(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    costs: dict,
    n: int = 100,
) -> pd.DataFrame:
    """
    Return the top-N most costly misclassifications.

    Parameters
    ----------
    X_test : pd.DataFrame — feature matrix for test set.
    y_test : array — true labels.
    y_pred : array — predicted labels.
    probs : array of shape (n_samples, 4) — calibrated probabilities.
    costs : dict — cost matrix config (from cost_matrix.yaml).
    n : int — number of worst cases to return.

    Returns
    -------
    pd.DataFrame with columns: actual, predicted, cost, max_prob, ...feature columns
    """
    y_true = np.array(y_test)
    y_pred = np.array(y_pred)

    # Compute per-sample cost
    sample_costs = np.array([
        costs.get(f"actual_{int(t)}", {}).get(f"pred_{int(p)}", 0.0)
        for t, p in zip(y_true, y_pred)
    ])

    # Only consider misclassified samples
    is_wrong = y_true != y_pred
    wrong_idx = np.where(is_wrong)[0]

    result_df = X_test.iloc[wrong_idx].copy()
    result_df["actual"] = y_true[wrong_idx]
    result_df["predicted"] = y_pred[wrong_idx]
    result_df["misclass_cost"] = sample_costs[wrong_idx]
    result_df["prob_actual"] = probs[wrong_idx, y_true[wrong_idx]]
    result_df["prob_predicted"] = probs[wrong_idx, y_pred[wrong_idx]]

    result_df = result_df.sort_values("misclass_cost", ascending=False).head(n)

    print(f"\n{'='*60}")
    print(f"  TOP {n} MISCLASSIFICATIONS (sorted by cost)")
    print(f"{'='*60}")
    print(f"Total misclassified: {int(is_wrong.sum())} / {len(y_true)}")
    print(
        result_df[["actual", "predicted", "misclass_cost", "prob_actual", "prob_predicted"]]
        .head(20)
        .to_string(index=False)
    )
    print(f"\n(Showing top 20 of {n}. Full result returned as DataFrame.)")

    # Summarize misclassification patterns
    print("\n  Misclassification pattern summary:")
    pattern_counts = (
        pd.DataFrame({"actual": y_true[wrong_idx], "predicted": y_pred[wrong_idx]})
        .groupby(["actual", "predicted"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    print(pattern_counts.to_string(index=False))
    print("=" * 60)

    return result_df
