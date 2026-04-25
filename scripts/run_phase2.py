"""Run Phase 2: Problem A Severity Model Training & Evaluation."""
import sys
import pandas as pd
import numpy as np

from src.utils.config import (
    load_main_config, load_feature_whitelist, load_cost_matrix,
    resolve_path, set_seeds,
)
from src.data.loader import load_raw_data, parse_time_date
from src.data.target_engineering import apply_severity_rubric, validate_severity_distribution
from src.data.leakage_audit import get_problem_a_features, validate_no_leakage
from src.features.temporal import (
    extract_temporal_features, create_temporal_split,
    validate_temporal_split, get_split_data,
)
from src.features.encoding import (
    identify_column_types, prepare_for_lgbm, bucket_experience,
)
from src.models.severity import (
    train_lgbm_severity, calibrate_model, predict_calibrated, save_model,
    train_catboost_severity,
)
from src.evaluation.ordinal_metrics import full_severity_report
from src.evaluation.calibration import expected_calibration_error

set_seeds()
config = load_main_config()

# ============================================================
# Load & prepare data
# ============================================================
print("Loading data...")
df = load_raw_data(config)
df = apply_severity_rubric(df)
df = parse_time_date(df)
df = create_temporal_split(df)
df = extract_temporal_features(df)
df = bucket_experience(df)

val = validate_severity_distribution(df)
print("Severity distribution:", val["distribution"])
print("Valid:", val["is_valid"])

# ============================================================
# Feature selection
# ============================================================
whitelist = load_feature_whitelist()
prob_a_feats = get_problem_a_features(df, whitelist)

# Add derived features
derived = ["year", "month", "quarter", "month_sin", "month_cos", "time_of_day_bucket"]
if "experience_bucket" in df.columns:
    derived.append("experience_bucket")
feature_cols = list(set(prob_a_feats + [d for d in derived if d in df.columns]))
feature_cols = [c for c in feature_cols if c != "Time_Date"]

is_clean, leaks = validate_no_leakage(feature_cols, problem="A", whitelist=whitelist)
print(f"\nFeature count: {len(feature_cols)}")
print(f"Leakage check: {'PASSED' if is_clean else 'FAILED'}")
if not is_clean:
    for l in leaks:
        print(f"  LEAK: {l}")
    sys.exit(1)

# ============================================================
# Prepare for LightGBM
# ============================================================
col_types = identify_column_types(df[feature_cols])
df = prepare_for_lgbm(df, col_types["categorical"], col_types["numeric"], col_types["medium_missing"])

# Drop high-missing columns
for hm in col_types["high_missing"]:
    if hm in feature_cols:
        feature_cols.remove(hm)
        print(f"Dropped (>80% missing): {hm}")

print(f"Final features: {len(feature_cols)}")
print(f"  Categoricals: {[c for c in col_types['categorical'] if c in feature_cols]}")

# ============================================================
# Split data
# ============================================================
splits = get_split_data(df, "severity_level", feature_cols)
X_train, y_train = splits["train"]
X_val, y_val = splits["val"]
X_test, y_test = splits["test"]
print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

cat_features = [c for c in col_types["categorical"] if c in feature_cols]

# ============================================================
# Train LightGBM
# ============================================================
print("\nTraining LightGBM severity model...")
lgb_model, lgb_history = train_lgbm_severity(X_train, y_train, X_val, y_val, cat_features)
print(f"Best iteration: {lgb_history['best_iteration']}")

print("Calibrating LightGBM...")
lgb_calibrators = calibrate_model(lgb_model, X_val, y_val)
lgb_cal_probs = predict_calibrated(lgb_model, X_test, lgb_calibrators)
lgb_y_pred = lgb_cal_probs.argmax(axis=1)

cost_config = load_cost_matrix()
lgb_report = full_severity_report(y_test.values, lgb_y_pred, cost_config.get("costs"))

# ============================================================
# Train CatBoost
# ============================================================
print("\nTraining CatBoost severity model...")
cb_model, X_train_cb, X_val_cb = train_catboost_severity(X_train, y_train, X_val, y_val, cat_features)

print("Calibrating CatBoost...")
cb_calibrators = calibrate_model(cb_model, X_val_cb, y_val)

X_test_cb = X_test.copy()
for col in cat_features:
    X_test_cb[col] = X_test_cb[col].astype(str).replace('nan', 'Missing').fillna('Missing')

cb_cal_probs = predict_calibrated(cb_model, X_test_cb, cb_calibrators)
cb_y_pred = cb_cal_probs.argmax(axis=1)
cb_report = full_severity_report(y_test.values, cb_y_pred, cost_config.get("costs"))

# ============================================================
# Compare & Select Best
# ============================================================
print("\n" + "="*50)
print("  MODEL COMPARISON (Test Set)")
print("="*50)
print(f"  LightGBM QWK: {lgb_report['qwk']:.4f}  (Macro-F1: {lgb_report['classification_report']['macro avg']['f1-score']:.4f})")
print(f"  CatBoost QWK: {cb_report['qwk']:.4f}  (Macro-F1: {cb_report['classification_report']['macro avg']['f1-score']:.4f})")

if cb_report['qwk'] > lgb_report['qwk']:
    print("\n  >> CatBoost wins. Saving CatBoost model...")
    best_model = cb_model
    best_report = cb_report
    best_probs = cb_cal_probs
    best_name = "CatBoost"
    model_path = resolve_path("models/severity_catboost.cbm")
    best_model.save_model(str(model_path))
else:
    print("\n  >> LightGBM wins. Saving LightGBM model...")
    best_model = lgb_model
    best_report = lgb_report
    best_probs = lgb_cal_probs
    best_name = "LightGBM"
    model_path = save_model(best_model, str(resolve_path("models")), "severity_lgbm")

print(f"  Model saved: {model_path}")

# ============================================================
# Best Model — Full Evaluation
# ============================================================
print("\n" + "="*50)
print(f"  BEST MODEL ({best_name}) EVALUATION")
print("="*50)
print(f"  QWK:          {best_report['qwk']:.4f}")
print(f"  QWK 95% CI:   [{best_report['qwk_bootstrap']['ci_low']:.4f}, {best_report['qwk_bootstrap']['ci_high']:.4f}]")
print(f"  Ordinal MAE:  {best_report['ordinal_mae']:.4f}")
if "asymmetric_cost" in best_report:
    print(f"  Asym. Cost:   {best_report['asymmetric_cost']:.4f}")

cls_report = best_report["classification_report"]
print("\n  Per-class metrics:")
for cls in ["0", "1", "2", "3"]:
    if cls in cls_report:
        m = cls_report[cls]
        print(f"    Level {cls}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1-score']:.3f} (n={m['support']})")

print(f"\n  Macro-F1:  {cls_report['macro avg']['f1-score']:.4f}")
print(f"  Weighted-F1: {cls_report['weighted avg']['f1-score']:.4f}")

print("\n  Calibration (ECE):")
for cls in range(4):
    ece = expected_calibration_error(
        (y_test == cls).astype(int).values, best_probs[:, cls]
    )
    print(f"    Class {cls}: {ece:.4f}")

print("\n  Confusion Matrix (rows=actual, cols=predicted):")
cm = best_report["confusion_matrix"]
print(f"  {'':>10} pred_0  pred_1  pred_2  pred_3")
for i, row in enumerate(cm):
    print(f"  actual_{i}  {row[0]:>6}  {row[1]:>6}  {row[2]:>6}  {row[3]:>6}")

# ============================================================
# Slice Analysis — Performance by Year
# ============================================================
print("\n" + "="*50)
print("  SLICE ANALYSIS & FEATURE IMPORTANCE")
print("="*50)

if "year" in X_test.columns:
    print("  Slice by Year:")
    X_slice = X_test_cb if best_name == "CatBoost" else X_test
    for yr in sorted(X_slice["year"].unique()):
        idx = X_slice["year"] == yr
        y_slice = y_test[idx]
        p_slice = best_probs[idx].argmax(axis=1)
        if len(y_slice) > 10:
            sr = full_severity_report(y_slice.values, p_slice, cost_config.get("costs"))
            print(f"    {int(yr)}: QWK={sr['qwk']:.4f}  Macro-F1={sr['classification_report']['macro avg']['f1-score']:.4f}  (n={len(y_slice)})")

# ============================================================
# Feature Importance (built-in, no SHAP/Numba required)
# ============================================================
print("\n  Feature Importance (top 10):")
if best_name == "LightGBM":
    importance = best_model.feature_importance(importance_type='gain')
else:
    importance = best_model.get_feature_importance()

top_idx = np.argsort(importance)[::-1][:10]
for i in top_idx:
    print(f"    {feature_cols[i]:<45} : {importance[i]:.2f}")

print("\nPhase 2 complete!")

