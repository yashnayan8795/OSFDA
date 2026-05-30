"""
Problem A — Incident Severity Classification Model
====================================================
LightGBM multiclass severity with Optuna, isotonic calibration, SHAP.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


def build_lgbm_dataset(X_train, y_train, X_val, y_val, categorical_features):
    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_features, reference=dtrain, free_raw_data=False)
    return dtrain, dval


def get_default_lgbm_params(num_classes=4, seed=42):
    return {
        "objective": "multiclass", "num_class": num_classes,
        "metric": "multi_logloss", "boosting_type": "gbdt",
        "num_leaves": 63, "max_depth": 8, "learning_rate": 0.05,
        "min_child_samples": 30, "colsample_bytree": 0.8,
        "subsample": 0.8, "subsample_freq": 5,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "seed": seed, "verbose": -1, "n_jobs": -1,
    }


def train_lgbm_severity(X_train, y_train, X_val, y_val, categorical_features,
                         params=None, num_boost_round=2000, early_stopping_rounds=100):
    if params is None:
        params = get_default_lgbm_params()
    dtrain, dval = build_lgbm_dataset(X_train, y_train, X_val, y_val, categorical_features)
    callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds), lgb.log_evaluation(period=100)]
    model = lgb.train(params, dtrain, num_boost_round=num_boost_round,
                      valid_sets=[dval], valid_names=["val"], callbacks=callbacks)
    return model, {"best_iteration": model.best_iteration, "best_score": model.best_score}


def train_catboost_severity(X_train, y_train, X_val, y_val, categorical_features):
    from catboost import CatBoostClassifier, Pool
    
    # Fill NA for categorical features in CatBoost
    X_train_cb = X_train.copy()
    X_val_cb = X_val.copy()
    for col in categorical_features:
        X_train_cb[col] = X_train_cb[col].astype(str).replace('nan', 'Missing').fillna('Missing')
        X_val_cb[col] = X_val_cb[col].astype(str).replace('nan', 'Missing').fillna('Missing')
        
    train_pool = Pool(X_train_cb, y_train, cat_features=categorical_features)
    val_pool = Pool(X_val_cb, y_val, cat_features=categorical_features)
    
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        random_seed=42,
        od_type='Iter',
        od_wait=50,
        verbose=100,
        task_type='CPU'
    )
    
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model, X_train_cb, X_val_cb


def train_catboost_severity_weighted(
    X_train, y_train, X_val, y_val, categorical_features,
    class_weights=None, seed: int = 42
):
    """
    Train CatBoost with explicit class weights for minority-class boosting.

    Parameters
    ----------
    class_weights : list of float, optional
        Per-class weights in label order [w0, w1, w2, w3].
        Example: [1, 5, 20, 100] heavily penalizes missing Level 3.
        If None, trains without class weighting (equivalent to train_catboost_severity).
    seed : int
        Random seed for reproducibility. Default 42.

    Returns
    -------
    (model, X_train_cb, X_val_cb)

    Notes
    -----
    After training with weights, you MUST rerun isotonic calibration + temperature
    scaling on the validation set before evaluating or accepting the model.
    The asymmetric loss shifts the model's probability estimates away from
    class priors; calibration corrects this shift.
    """
    from catboost import CatBoostClassifier, Pool

    X_train_cb = X_train.copy()
    X_val_cb = X_val.copy()
    for col in categorical_features:
        if col in X_train_cb.columns:
            X_train_cb[col] = X_train_cb[col].astype(str).replace("nan", "Missing").fillna("Missing")
            X_val_cb[col] = X_val_cb[col].astype(str).replace("nan", "Missing").fillna("Missing")

    train_pool = Pool(X_train_cb, y_train, cat_features=categorical_features)
    val_pool = Pool(X_val_cb, y_val, cat_features=categorical_features)

    model_params = dict(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=seed,
        od_type="Iter",
        od_wait=50,
        verbose=100,
        task_type="CPU",
    )
    if class_weights is not None:
        model_params["class_weights"] = class_weights

    model = CatBoostClassifier(**model_params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model, X_train_cb, X_val_cb




def optuna_objective(trial, X_train, y_train, X_val, y_val, categorical_features):
    params = {
        "objective": "multiclass", "num_class": 4, "metric": "multi_logloss",
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "verbose": -1, "n_jobs": -1,
    }
    dtrain, dval = build_lgbm_dataset(X_train, y_train, X_val, y_val, categorical_features)
    callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
    model = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dval], valid_names=["val"], callbacks=callbacks)
    return model.best_score["val"]["multi_logloss"]


def calibrate_model(model, X_val, y_val, num_classes=4):
    from sklearn.isotonic import IsotonicRegression
    if hasattr(model, 'predict_proba'):
        raw_probs = model.predict_proba(X_val)
    else:
        raw_probs = model.predict(X_val)
    calibrators = []
    for cls in range(num_classes):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_probs[:, cls], (y_val == cls).astype(int))
        calibrators.append(iso)
    return calibrators


def predict_calibrated(model, X, calibrators):
    if hasattr(model, 'predict_proba'):
        raw_probs = model.predict_proba(X)
    else:
        raw_probs = model.predict(X)
    cal_probs = np.column_stack([cal.predict(raw_probs[:, i]) for i, cal in enumerate(calibrators)])
    row_sums = cal_probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return cal_probs / row_sums


def build_cost_matrix(cost_config: Dict[str, Any], num_classes: int = 4) -> np.ndarray:
    """
    Convert the cost_matrix.yaml ``costs`` dict into a numpy array C where
    C[actual, predicted] = cost.  Used by predict_cost_sensitive().
    """
    C = np.zeros((num_classes, num_classes), dtype=float)
    for i in range(num_classes):
        row_key = f"actual_{i}"
        row = cost_config.get(row_key, {})
        for j in range(num_classes):
            C[i, j] = row.get(f"pred_{j}", 0.0)
    return C


def predict_cost_sensitive(
    probs: np.ndarray,
    cost_config: Dict[str, Any],
    num_classes: int = 4,
) -> np.ndarray:
    """
    Choose the prediction that minimises expected cost for each sample.

    E[cost | pred=j] = sum_i P(actual=i) * C[i, j]

    Parameters
    ----------
    probs : np.ndarray, shape (n_samples, num_classes)
        Calibrated class probabilities.
    cost_config : dict
        The ``costs`` sub-dict from cost_matrix.yaml.

    Returns
    -------
    np.ndarray of shape (n_samples,) — integer class predictions.
    """
    C = build_cost_matrix(cost_config, num_classes)  # shape (num_classes, num_classes)
    # expected_cost[sample, j] = probs[sample, :] @ C[:, j]
    expected_cost = probs @ C           # (n_samples, num_classes)
    return expected_cost.argmin(axis=1)


def compute_shap_values(model, X):
    import shap
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X)


def save_model(model, save_dir="models", name="severity_lgbm"):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    model_file = save_path / f"{name}.txt"
    model.save_model(str(model_file))
    return model_file
