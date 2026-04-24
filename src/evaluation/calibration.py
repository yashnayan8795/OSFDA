"""
Calibration Diagnostics
========================
Reliability diagrams, ECE, per-class calibration curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def expected_calibration_error(y_true, y_prob, n_bins=10) -> float:
    """Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() * np.abs(bin_acc - bin_conf)
    return ece / len(y_true)


def plot_reliability_diagram(y_true, y_prob, n_bins=10, class_name="", ax=None):
    """Plot a reliability diagram for a single class."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs = [], []
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_accs.append(y_true[mask].mean())
        bin_confs.append(y_prob[mask].mean())
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.bar(bin_confs, bin_accs, width=1.0 / n_bins, alpha=0.6, edgecolor="black")
    ece = expected_calibration_error(y_true, y_prob, n_bins)
    ax.set_title(f"Reliability — {class_name} (ECE={ece:.4f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.legend()
    return ax
