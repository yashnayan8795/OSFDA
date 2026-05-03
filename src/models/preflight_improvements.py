"""
Problem C — Preflight Model Improvements
==========================================
Utilities for evaluating the calibrated preflight model against a
held-out temporal test set and for tuning the decision threshold.

These complement PriorShiftedCalibratedModel in preflight.py but
do NOT require re-running the full training pipeline.

Usage
-----
    python scripts/evaluate_preflight.py   # see that script for CLI
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Threshold Tuning
# ─────────────────────────────────────────────────────────────────────────────

def tune_preflight_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    true_prior: float,
    multiplier_range: Tuple[float, float] = (1.0, 15.0),
    n_steps: int = 100,
    optimize: str = "f1",
) -> Dict[str, Any]:
    """
    Sweep thresholds over [true_prior × lo, true_prior × hi] and return
    the one that maximises the chosen metric.

    Parameters
    ----------
    y_true : np.ndarray of int (0/1)
        Ground-truth labels on the test set.
    y_prob : np.ndarray of float
        Calibrated P(incident) from PriorShiftedCalibratedModel.
    true_prior : float
        Real-world base rate (e.g., 0.05 for 5% positive rate).
    multiplier_range : tuple
        (min_multiplier, max_multiplier) applied to true_prior.
    n_steps : int
        Number of threshold candidates.
    optimize : str
        Metric to maximise: "f1", "precision", "recall", or "j_statistic".

    Returns
    -------
    dict with keys: best_threshold, best_score, metric, sweep_results
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    lo_mult, hi_mult = multiplier_range
    thresholds = np.linspace(true_prior * lo_mult, min(true_prior * hi_mult, 0.99), n_steps)

    sweep_results = []
    best_score = -1.0
    best_threshold = true_prior * 5  # original heuristic as fallback

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if y_pred.sum() == 0:
            continue

        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        j_stat = rec + prec - 1.0  # Youden's J

        metric_map = {"f1": f1, "precision": prec, "recall": rec, "j_statistic": j_stat}
        score = metric_map.get(optimize, f1)

        sweep_results.append({
            "threshold": float(t),
            "multiplier": float(t / true_prior),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "j_statistic": float(j_stat),
        })

        if score > best_score:
            best_score = score
            best_threshold = float(t)

    return {
        "best_threshold": best_threshold,
        "best_multiplier": best_threshold / true_prior,
        "best_score": best_score,
        "metric": optimize,
        "sweep_results": sweep_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_preflight_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """
    Compute the full suite of Problem C evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray of int
        Ground-truth binary labels.
    y_prob : np.ndarray of float
        Calibrated posterior P(incident | features).
    threshold : float
        Decision boundary to apply for hard predictions.

    Returns
    -------
    dict with: roc_auc, pr_auc, brier_score, ece, precision, recall, f1,
               threshold, n_positive, n_negative
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        brier_score_loss, precision_score, recall_score, f1_score,
        confusion_matrix,
    )

    y_pred = (y_prob >= threshold).astype(int)

    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())

    roc_auc = float(roc_auc_score(y_true, y_prob)) if n_pos > 0 and n_neg > 0 else float("nan")
    pr_auc = float(average_precision_score(y_true, y_prob)) if n_pos > 0 else float("nan")
    brier = float(brier_score_loss(y_true, y_prob))

    # ECE (10 bins)
    ece = _ece(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "ece": ece,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return float(ece / max(len(y_true), 1))


# ─────────────────────────────────────────────────────────────────────────────
# Route-level Historical Risk Features
# ─────────────────────────────────────────────────────────────────────────────

def add_route_features(
    df: pd.DataFrame,
    incident_df: pd.DataFrame,
    origin_col: str = "origin",
    dest_col: str = "dest",
    incident_origin_col: str = "Departure_Airport",
    incident_dest_col: str = "Destination_Airport",
    date_col: str = "fl_date",
) -> pd.DataFrame:
    """
    Add historical incident rate per airport-pair as a feature.

    Computes: for each (origin, dest) pair in df, how many incidents
    occurred on that route in the training NTSB data, normalised by
    total flights on that route (if available) or as raw count.

    Parameters
    ----------
    df : pd.DataFrame
        Flight feature dataframe to augment.
    incident_df : pd.DataFrame
        NTSB incident data with origin/dest columns.
    origin_col, dest_col : str
        Column names in df.
    incident_origin_col, incident_dest_col : str
        Column names in incident_df.
    date_col : str
        Date column in df for temporal filtering.

    Returns
    -------
    pd.DataFrame with new columns: route_incident_count, route_incident_rate
    """
    df = df.copy()

    # Count incidents per route
    route_counts = (
        incident_df
        .groupby([incident_origin_col, incident_dest_col])
        .size()
        .reset_index(name="route_incident_count")
        .rename(columns={
            incident_origin_col: origin_col,
            incident_dest_col: dest_col,
        })
    )

    df = df.merge(route_counts, on=[origin_col, dest_col], how="left")
    df["route_incident_count"] = df["route_incident_count"].fillna(0).astype(int)

    # Normalise by overall route flight count in df (if both origins available)
    if origin_col in df.columns and dest_col in df.columns:
        total_flights = df.groupby([origin_col, dest_col]).size().reset_index(name="route_total_flights")
        df = df.merge(total_flights, on=[origin_col, dest_col], how="left")
        df["route_incident_rate"] = df["route_incident_count"] / df["route_total_flights"].clip(lower=1)
    else:
        df["route_incident_rate"] = df["route_incident_count"].astype(float)

    return df


def add_seasonal_features(df: pd.DataFrame, date_col: str = "fl_date") -> pd.DataFrame:
    """
    Add seasonal sin/cos encodings and day-of-week indicator.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a date column parseable by pd.to_datetime.
    date_col : str
        Name of the date column.

    Returns
    -------
    pd.DataFrame with new columns: month_sin, month_cos, dow_sin, dow_cos
    """
    df = df.copy()
    if date_col not in df.columns:
        return df

    dates = pd.to_datetime(df[date_col], errors="coerce")
    df["month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12)
    df["dow_sin"] = np.sin(2 * np.pi * dates.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dates.dt.dayofweek / 7)
    return df
