"""Run Optuna hyperparameter tuning for the severity model."""
import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb

from src.utils.config import (
    load_main_config, load_feature_whitelist, load_cost_matrix,
    resolve_path, set_seeds,
)
from src.data.loader import load_raw_data, parse_time_date
from src.data.target_engineering import apply_severity_rubric
from src.data.leakage_audit import get_problem_a_features
from src.features.temporal import (
    extract_temporal_features, create_temporal_split, get_split_data,
)
from src.features.encoding import (
    identify_column_types, prepare_for_lgbm, bucket_experience,
)
from src.models.severity import (
    build_lgbm_dataset, train_lgbm_severity, calibrate_model,
    predict_calibrated, save_model,
)
from src.evaluation.ordinal_metrics import full_severity_report, quadratic_weighted_kappa
from src.evaluation.calibration import expected_calibration_error

set_seeds()
config = load_main_config()

# Load & prepare
print("Loading and preparing data...")
df = load_raw_data(config)
df = apply_severity_rubric(df)
df = parse_time_date(df)
df = create_temporal_split(df)
df = extract_temporal_features(df)
df = bucket_experience(df)

whitelist = load_feature_whitelist()
prob_a_feats = get_problem_a_features(df, whitelist)
derived = ["year", "month", "quarter", "month_sin", "month_cos", "time_of_day_bucket"]
if "experience_bucket" in df.columns:
    derived.append("experience_bucket")
feature_cols = list(set(prob_a_feats + [d for d in derived if d in df.columns]))
feature_cols = [c for c in feature_cols if c != "Time_Date"]

col_types = identify_column_types(df[feature_cols])
df = prepare_for_lgbm(df, col_types["categorical"], col_types["numeric"], col_types["medium_missing"])
for hm in col_types["high_missing"]:
    if hm in feature_cols:
        feature_cols.remove(hm)

splits = get_split_data(df, "severity_level", feature_cols)
X_train, y_train = splits["train"]
X_val, y_val = splits["val"]
X_test, y_test = splits["test"]
cat_features = [c for c in col_types["categorical"] if c in feature_cols]

print(f"Features: {len(feature_cols)}, Train: {len(X_train)}, Val: {len(X_val)}")

# ============================================================
# Optuna study
# ============================================================
def objective(trial):
    params = {
        "objective": "multiclass",
        "num_class": 4,
        "metric": "multi_logloss",
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

    dtrain, dval = build_lgbm_dataset(X_train, y_train, X_val, y_val, cat_features)
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=0),
    ]
    model = lgb.train(
        params, dtrain, num_boost_round=2000,
        valid_sets=[dval], valid_names=["val"],
        callbacks=callbacks,
    )

    # Evaluate with QWK (our real metric)
    val_probs = model.predict(X_val)
    val_pred = val_probs.argmax(axis=1)
    qwk = quadratic_weighted_kappa(y_val, val_pred)

    return qwk  # Maximize


storage_path = resolve_path("data/processed/optuna.db")
storage_path.parent.mkdir(parents=True, exist_ok=True)
storage_url = f"sqlite:///{storage_path}"
study = optuna.create_study(
    direction="maximize",
    study_name="severity_lgbm",
    storage=storage_url,
    load_if_exists=True,
)
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\nBest QWK (val): {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# ============================================================
# Retrain with best params on full train data
# ============================================================
print("\nRetraining with best params...")
best_params = {
    "objective": "multiclass",
    "num_class": 4,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}
best_params.update(study.best_params)

model, history = train_lgbm_severity(
    X_train, y_train, X_val, y_val, cat_features,
    params=best_params, num_boost_round=3000, early_stopping_rounds=150,
)
print(f"Best iteration: {history['best_iteration']}")

# Calibrate & evaluate
calibrators = calibrate_model(model, X_val, y_val)
cal_probs = predict_calibrated(model, X_test, calibrators)
y_pred = cal_probs.argmax(axis=1)

cost_config = load_cost_matrix()
report = full_severity_report(y_test.values, y_pred, cost_config.get("costs"))

print("\n" + "="*50)
print("  TUNED MODEL — TEST SET EVALUATION")
print("="*50)
print(f"  QWK:          {report['qwk']:.4f}")
print(f"  QWK 95% CI:   [{report['qwk_bootstrap']['ci_low']:.4f}, {report['qwk_bootstrap']['ci_high']:.4f}]")
print(f"  Ordinal MAE:  {report['ordinal_mae']:.4f}")
if "asymmetric_cost" in report:
    print(f"  Asym. Cost:   {report['asymmetric_cost']:.4f}")

cls_report = report["classification_report"]
print(f"\n  Macro-F1:     {cls_report['macro avg']['f1-score']:.4f}")
print(f"  Weighted-F1:  {cls_report['weighted avg']['f1-score']:.4f}")

for cls in ["0", "1", "2", "3"]:
    if cls in cls_report:
        m = cls_report[cls]
        print(f"    Level {cls}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1-score']:.3f}")

# Save tuned model
model_path = save_model(model, str(resolve_path("models")), "severity_lgbm_tuned")
print(f"\nTuned model saved: {model_path}")
