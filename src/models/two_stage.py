"""
Two-Stage Severity Classifier — Problem A
==========================================
Stage 1: Binary CatBoost  — P(S >= 1)  [Minor vs. Not-Minor]
Stage 2: Ordinal CatBoost — P(S | S >= 1) [among non-minor only]

Joint probability reconstruction:
    P(S=0)   = 1 - P(S >= 1)
    P(S=s)   = P(S >= 1) * P(S=s | S >= 1)   for s in {1, 2, 3}

Calibration pipeline:
    1. Stage 1 binary: isotonic calibration on full val set.
    2. Stage 2 ordinal: isotonic + temperature scaling on val-subset (y >= 1).
    3. Joint probs: temperature scaling on full val set (after reconstruction).

Temporal integrity:
    - All calibration fitted exclusively on validation split (2018-2019).
    - Test set (2019-2022) is never used during any fitting step.

Seeds:
    - All CatBoost models use random_seed=42.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.isotonic import IsotonicRegression

from src.models.ordinal import (
    train_ordinal_catboost,
    predict_ordinal_catboost_probs,
    calibrate_ordinal_catboost,
    predict_ordinal_catboost_calibrated,
    temperature_scale,
    apply_temperature,
)


# ---------------------------------------------------------------------------
# Stage 1: Binary Minor / Not-Minor
# ---------------------------------------------------------------------------

def _catboost_binary_params(seed: int = 42) -> dict:
    return {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "random_seed": seed,
        "od_type": "Iter",
        "od_wait": 50,
        "verbose": 0,
        "task_type": "CPU",
    }


def _prepare_catboost_df(df: pd.DataFrame, categorical_features: List[str]) -> pd.DataFrame:
    """CatBoost-safe DataFrame: categoricals as string, NaN -> 'Missing'."""
    df = df.copy()
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", "Missing").fillna("Missing")
    return df


def train_stage1(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    categorical_features: List[str],
    seed: int = 42,
):
    """
    Train Stage 1: binary classifier for P(S >= 1) on the full dataset.

    Parameters
    ----------
    X_train, X_val : pd.DataFrame
    y_train, y_val : np.ndarray — severity labels in {0,1,2,3}
    categorical_features : list of str
    seed : int

    Returns
    -------
    (model, X_train_cb, X_val_cb)
    """
    from catboost import CatBoostClassifier, Pool

    # Binary target: 0 = Minor, 1 = Not-Minor
    y_train_bin = (np.array(y_train) >= 1).astype(int)
    y_val_bin = (np.array(y_val) >= 1).astype(int)

    X_train_cb = _prepare_catboost_df(X_train, categorical_features)
    X_val_cb = _prepare_catboost_df(X_val, categorical_features)

    train_pool = Pool(X_train_cb, y_train_bin, cat_features=categorical_features)
    val_pool = Pool(X_val_cb, y_val_bin, cat_features=categorical_features)

    model = CatBoostClassifier(**_catboost_binary_params(seed))
    print("  [TwoStage] Training Stage 1: P(S>=1) binary classifier...")
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    print(f"  [TwoStage] Stage 1 best iteration: {model.best_iteration_}")
    return model, X_train_cb, X_val_cb


# ---------------------------------------------------------------------------
# Stage 2: Ordinal CatBoost on Non-Minor subset
# ---------------------------------------------------------------------------

def train_stage2(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    categorical_features: List[str],
    seed: int = 42,
):
    """
    Train Stage 2: ordinal cumulative CatBoost on severity >= 1 subset.

    Labels are remapped: {1->0, 2->1, 3->2} for the 3-class ordinal model.
    Original labels {1,2,3} are reconstructed during joint probability computation.

    Returns
    -------
    (classifiers, info, X_train_sub_cb, X_val_sub_cb)
    """
    y_train_arr = np.array(y_train)
    y_val_arr = np.array(y_val)

    # Subset to non-minor
    train_mask = y_train_arr >= 1
    val_mask = y_val_arr >= 1

    X_train_sub = X_train[train_mask].copy()
    y_train_sub = y_train_arr[train_mask] - 1   # remap: 1->0, 2->1, 3->2
    X_val_sub = X_val[val_mask].copy()
    y_val_sub = y_val_arr[val_mask] - 1

    n_train_sub = int(train_mask.sum())
    n_val_sub = int(val_mask.sum())
    print(f"  [TwoStage] Stage 2 subset: train={n_train_sub}, val={n_val_sub}")

    if n_val_sub < 10:
        raise ValueError(
            f"Too few val samples with severity>=1 ({n_val_sub}). "
            "Cannot reliably train or calibrate Stage 2."
        )

    # Train ordinal CatBoost on 3-class target
    classifiers, info = train_ordinal_catboost(
        X_train_sub, y_train_sub,
        X_val_sub, y_val_sub,
        categorical_features,
        num_classes=3,
        seed=seed,
    )
    return classifiers, info, X_train_sub, X_val_sub, y_train_sub, y_val_sub


# ---------------------------------------------------------------------------
# Joint Probability Reconstruction
# ---------------------------------------------------------------------------

def reconstruct_joint_probs(
    stage1_p_not_minor: np.ndarray,   # shape (n,)
    stage2_probs_conditional: np.ndarray,  # shape (n, 3) — P(S=s | S>=1) for s=1,2,3
) -> np.ndarray:
    """
    Reconstruct 4-class joint probabilities from stage outputs.

    P(S=0)   = 1 - P(S>=1)
    P(S=s)   = P(S>=1) * P(S=s | S>=1)   for s in {1,2,3}

    Parameters
    ----------
    stage1_p_not_minor : np.ndarray, shape (n,) — P(S>=1) from Stage 1.
    stage2_probs_conditional : np.ndarray, shape (n, 3) — P(S=1|S>=1), P(S=2|S>=1), P(S=3|S>=1).

    Returns
    -------
    np.ndarray of shape (n, 4)
    """
    n = len(stage1_p_not_minor)
    joint = np.zeros((n, 4), dtype=np.float64)

    joint[:, 0] = 1.0 - stage1_p_not_minor
    for s in range(3):  # s=0,1,2 in conditional → s=1,2,3 in joint
        joint[:, s + 1] = stage1_p_not_minor * stage2_probs_conditional[:, s]

    # Numerical safety: clip and renormalise
    joint = np.clip(joint, 0, None)
    row_sums = joint.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return joint / row_sums


# ---------------------------------------------------------------------------
# Full Two-Stage Training + Calibration
# ---------------------------------------------------------------------------

class TwoStageSeverityModel:
    """
    End-to-end two-stage severity classifier with calibration.

    Attributes
    ----------
    stage1_model : CatBoostClassifier
    stage2_classifiers : list of CatBoostClassifier (3 ordinal thresholds)
    stage1_calibrator : IsotonicRegression (for P(S>=1))
    stage2_calibrators : list of IsotonicRegression (per conditional class)
    joint_temperature : float (temperature for joint probability scaling)
    categorical_features : list of str
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.stage1_model = None
        self.stage2_classifiers = None
        self.stage1_calibrator = None
        self.stage2_calibrators = None
        self.joint_temperature = 1.0
        self.categorical_features = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        categorical_features: List[str],
    ) -> "TwoStageSeverityModel":
        """
        Train both stages and perform all calibration on the validation set.

        Parameters
        ----------
        All data must be from train split (2012-2018) and val split (2018-2019).
        Test split (2019-2022) must NEVER be passed to this method.
        """
        self.categorical_features = categorical_features
        y_train_arr = np.array(y_train)
        y_val_arr = np.array(y_val)

        # -- Stage 1 --
        self.stage1_model, X_train_cb, X_val_cb = train_stage1(
            X_train, y_train_arr, X_val, y_val_arr, categorical_features, self.seed
        )

        # Calibrate Stage 1 binary probability on val
        raw_s1_val = self.stage1_model.predict_proba(X_val_cb)[:, 1]
        self.stage1_calibrator = IsotonicRegression(out_of_bounds="clip")
        self.stage1_calibrator.fit(raw_s1_val, (y_val_arr >= 1).astype(int))

        # -- Stage 2 --
        (
            self.stage2_classifiers, _,
            X_train_sub, X_val_sub, y_train_sub, y_val_sub,
        ) = train_stage2(
            X_train, y_train_arr, X_val, y_val_arr, categorical_features, self.seed
        )

        # Calibrate Stage 2 ordinal on val subset
        self.stage2_calibrators = calibrate_ordinal_catboost(
            self.stage2_classifiers, X_val_sub, y_val_sub,
            categorical_features, num_classes=3,
        )

        # -- Joint calibration with temperature scaling on full val --
        joint_val = self._predict_joint(X_val, y_val_arr)
        _, self.joint_temperature = temperature_scale(joint_val, y_val_arr, num_classes=4)
        print(f"  [TwoStage] Joint temperature: {self.joint_temperature:.3f}")

        return self

    def _predict_joint(
        self, X: pd.DataFrame, y_true_hint: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Internal: return raw joint probabilities (before joint temperature)."""
        X_cb = _prepare_catboost_df(X, self.categorical_features)

        # Stage 1: P(S >= 1) — calibrated
        raw_s1 = self.stage1_model.predict_proba(X_cb)[:, 1]
        cal_s1 = np.clip(self.stage1_calibrator.predict(raw_s1), 0, 1)

        # Stage 2: P(S | S >= 1) — calibrated ordinal
        cal_s2 = predict_ordinal_catboost_calibrated(
            self.stage2_classifiers, X,
            self.stage2_calibrators, self.categorical_features, num_classes=3,
        )

        joint = reconstruct_joint_probs(cal_s1, cal_s2)
        return joint

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return joint temperature-scaled probabilities, shape (n, 4).

        Use for calibration evaluation and cost-sensitive decoding.
        """
        joint = self._predict_joint(X)
        return apply_temperature(joint, self.joint_temperature)

    def predict(self, X: pd.DataFrame, costs: Optional[dict] = None) -> np.ndarray:
        """
        Return integer class predictions.

        If costs is provided, uses cost-minimising decoding.
        Otherwise returns argmax of probabilities.
        """
        probs = self.predict_proba(X)
        if costs is not None:
            from src.models.severity import predict_cost_sensitive
            return predict_cost_sensitive(probs, costs)
        return probs.argmax(axis=1)
