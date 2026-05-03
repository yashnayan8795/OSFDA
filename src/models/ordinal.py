"""
Ordinal Cumulative Logit Decomposition — Problem A
====================================================
Trains K-1 binary LightGBM classifiers for a K-class ordinal target:

    clf_k fits:  P(Y >= k)  for k in {1, 2, ..., K-1}

Class probabilities are reconstructed as:
    P(Y = 0) = 1 - P(Y >= 1)
    P(Y = k) = P(Y >= k) - P(Y >= k+1)   for 0 < k < K-1
    P(Y = K-1) = P(Y >= K-1)

This is natively QWK-compatible because misclassifications that are
further apart in the ordinal scale have geometrically higher cost.

Temperature Scaling (post-isotonic)
-------------------------------------
After isotonic calibration a single temperature τ is learned on the
validation set that minimises NLL.  Dividing logits by τ > 1 reduces
overconfidence; τ < 1 sharpens under-confident models.

Usage
-----
    from src.models.ordinal import (
        train_ordinal_lgbm, predict_ordinal, calibrate_ordinal,
        predict_ordinal_calibrated, temperature_scale,
    )
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, List, Optional, Tuple
from sklearn.isotonic import IsotonicRegression


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _get_ordinal_params(num_leaves=63, seed=42, extra: Optional[Dict] = None) -> Dict[str, Any]:
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": num_leaves,
        "max_depth": 8,
        "learning_rate": 0.05,
        "min_child_samples": 30,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "subsample_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "seed": seed,
        "verbose": -1,
        "n_jobs": -1,
    }
    if extra:
        params.update(extra)
    return params


def train_ordinal_lgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    categorical_features: List[str],
    num_classes: int = 4,
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 100,
) -> Tuple[List[lgb.Booster], Dict[str, Any]]:
    """
    Train K-1 binary LightGBM classifiers for ordinal regression.

    Parameters
    ----------
    X_train, X_val : pd.DataFrame
        Feature matrices.
    y_train, y_val : np.ndarray
        Integer class labels in {0, 1, ..., num_classes-1}.
    categorical_features : list of str
        Column names treated as categoricals by LightGBM.
    num_classes : int
        Number of ordinal levels (default 4 for Problem A).
    params : dict, optional
        LightGBM binary params. Defaults to _get_ordinal_params().
    num_boost_round : int
        Maximum boosting rounds per threshold classifier.
    early_stopping_rounds : int
        Early stopping patience.

    Returns
    -------
    (classifiers, info)
        classifiers : list of lgb.Booster of length K-1
            classifiers[k] ≈ P(Y >= k+1)
        info : dict with keys 'best_iterations', 'thresholds'
    """
    if params is None:
        params = _get_ordinal_params()

    classifiers = []
    best_iterations = []

    for k in range(1, num_classes):
        y_bin_train = (y_train >= k).astype(int)
        y_bin_val   = (y_val   >= k).astype(int)

        dtrain = lgb.Dataset(
            X_train, label=y_bin_train,
            categorical_feature=categorical_features,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X_val, label=y_bin_val,
            categorical_feature=categorical_features,
            reference=dtrain,
            free_raw_data=False,
        )
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=100),
        ]
        print(f"  [Ordinal] Training threshold clf P(Y>={k})...")
        clf = lgb.train(
            params, dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=callbacks,
        )
        classifiers.append(clf)
        best_iterations.append(clf.best_iteration)

    info = {
        "best_iterations": best_iterations,
        "thresholds": list(range(1, num_classes)),
    }
    return classifiers, info


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_ordinal_probs(
    classifiers: List[lgb.Booster],
    X: pd.DataFrame,
    num_classes: int = 4,
) -> np.ndarray:
    """
    Reconstruct class probabilities from K-1 threshold classifiers.

    Returns
    -------
    np.ndarray of shape (n_samples, num_classes)
        Each row sums to 1.
    """
    # cum_probs[k] = P(Y >= k+1) for k in 0..K-2
    cum_probs = np.column_stack([clf.predict(X) for clf in classifiers])  # (n, K-1)

    # Clip to valid probability range
    cum_probs = np.clip(cum_probs, 1e-7, 1 - 1e-7)

    # Enforce monotonicity: P(Y>=1) >= P(Y>=2) >= P(Y>=3)
    for j in range(1, cum_probs.shape[1]):
        cum_probs[:, j] = np.minimum(cum_probs[:, j], cum_probs[:, j - 1])

    # P(Y=0) = 1 - P(Y>=1)
    # P(Y=k) = P(Y>=k) - P(Y>=k+1)  for 0 < k < K-1
    # P(Y=K-1) = P(Y>=K-1)
    class_probs = np.zeros((len(X), num_classes), dtype=np.float64)
    class_probs[:, 0] = 1.0 - cum_probs[:, 0]
    for k in range(1, num_classes - 1):
        class_probs[:, k] = cum_probs[:, k - 1] - cum_probs[:, k]
    class_probs[:, num_classes - 1] = cum_probs[:, num_classes - 2]

    # Clip and renormalise (numerical safety)
    class_probs = np.clip(class_probs, 0, None)
    row_sums = class_probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return class_probs / row_sums


def predict_ordinal(
    classifiers: List[lgb.Booster],
    X: pd.DataFrame,
    num_classes: int = 4,
) -> np.ndarray:
    """Return integer class predictions (argmax of reconstructed probs)."""
    probs = predict_ordinal_probs(classifiers, X, num_classes)
    return probs.argmax(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration: Isotonic + Temperature Scaling
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_ordinal(
    classifiers: List[lgb.Booster],
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    num_classes: int = 4,
) -> List[IsotonicRegression]:
    """
    Fit per-class isotonic regressors on validation set probabilities.

    Returns a list of IsotonicRegression objects (one per class).
    """
    probs = predict_ordinal_probs(classifiers, X_val, num_classes)
    calibrators = []
    for cls in range(num_classes):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(probs[:, cls], (y_val == cls).astype(int))
        calibrators.append(iso)
    return calibrators


def predict_ordinal_calibrated(
    classifiers: List[lgb.Booster],
    X: pd.DataFrame,
    calibrators: List[IsotonicRegression],
    num_classes: int = 4,
) -> np.ndarray:
    """
    Return calibrated class probabilities after isotonic correction.
    """
    raw_probs = predict_ordinal_probs(classifiers, X, num_classes)
    cal_probs = np.column_stack([
        cal.predict(raw_probs[:, i]) for i, cal in enumerate(calibrators)
    ])
    cal_probs = np.clip(cal_probs, 0, None)
    row_sums = cal_probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return cal_probs / row_sums


def temperature_scale(
    probs: np.ndarray,
    y_val: np.ndarray,
    num_classes: int = 4,
    n_steps: int = 100,
    t_range: Tuple[float, float] = (0.1, 10.0),
) -> Tuple[np.ndarray, float]:
    """
    Find the scalar temperature τ that minimises NLL on the validation set,
    then return temperature-scaled probabilities.

    Logits are approximated as log(p / (1-p)); temperature divides logits.
    This is applied after isotonic calibration as a final ECE reduction step.

    Parameters
    ----------
    probs : np.ndarray, shape (n_samples, num_classes)
        Calibrated class probabilities (after isotonic).
    y_val : np.ndarray
        Integer ground-truth labels.
    n_steps : int
        Number of temperature candidates to try.
    t_range : tuple
        (min_temperature, max_temperature) to sweep.

    Returns
    -------
    (scaled_probs, best_temperature)
    """
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    logits = np.log(probs / (1.0 - probs))  # approx inverse-sigmoid per class

    best_nll = float("inf")
    best_t = 1.0
    temperatures = np.linspace(t_range[0], t_range[1], n_steps)

    for t in temperatures:
        scaled_logits = logits / t
        # Softmax over classes
        exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
        scaled_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        # NLL
        nll = -np.mean(np.log(scaled_probs[np.arange(len(y_val)), y_val] + 1e-15))
        if nll < best_nll:
            best_nll = nll
            best_t = t

    # Apply best temperature
    scaled_logits = logits / best_t
    exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
    best_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return best_probs, best_t


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """
    Apply a pre-fitted temperature to a probability matrix without refitting.

    Use this to scale test-set probabilities using the temperature that was
    learned on the validation set via temperature_scale().

    Parameters
    ----------
    probs : np.ndarray, shape (n_samples, num_classes)
        Calibrated class probabilities to rescale.
    temperature : float
        Temperature value returned by temperature_scale().

    Returns
    -------
    np.ndarray of same shape, renormalised.
    """
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    logits = np.log(probs / (1.0 - probs))
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


# ─────────────────────────────────────────────────────────────────────────────
# Optuna Objective for Ordinal Model
# ─────────────────────────────────────────────────────────────────────────────

def ordinal_optuna_objective(
    trial,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    categorical_features: List[str],
    num_classes: int = 4,
):
    """
    Optuna objective for the ordinal LightGBM model.
    Maximises QWK on the validation set.
    """
    from sklearn.metrics import cohen_kappa_score

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 100.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 100.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 5.0),
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    try:
        clfs, _ = train_ordinal_lgbm(
            X_train, y_train, X_val, y_val,
            categorical_features, num_classes=num_classes,
            params=params, num_boost_round=1000, early_stopping_rounds=50,
        )
        val_pred = predict_ordinal(clfs, X_val, num_classes)
        qwk = cohen_kappa_score(y_val, val_pred, weights="quadratic")
        return qwk
    except Exception as e:
        print(f"  [Ordinal Optuna] Trial failed: {e}")
        return -1.0
