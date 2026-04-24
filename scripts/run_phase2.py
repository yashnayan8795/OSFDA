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
model, history = train_lgbm_severity(X_train, y_train, X_val, y_val, cat_features)
print(f"Best iteration: {history['best_iteration']}")

# ============================================================
# Calibration
# ============================================================
print("\nCalibrating...")
calibrators = calibrate_model(model, X_val, y_val)

# ============================================================
# Evaluate on test set
# ============================================================
print("\n" + "="*50)
print("  TEST SET EVALUATION")
print("="*50)

cal_probs = predict_calibrated(model, X_test, calibrators)
y_pred = cal_probs.argmax(axis=1)

cost_config = load_cost_matrix()
report = full_severity_report(y_test.values, y_pred, cost_config.get("costs"))

print(f"  QWK:          {report['qwk']:.4f}")
print(f"  QWK 95% CI:   [{report['qwk_bootstrap']['ci_low']:.4f}, {report['qwk_bootstrap']['ci_high']:.4f}]")
print(f"  Ordinal MAE:  {report['ordinal_mae']:.4f}")
if "asymmetric_cost" in report:
    print(f"  Asym. Cost:   {report['asymmetric_cost']:.4f}")

# Per-class metrics
cls_report = report["classification_report"]
print("\n  Per-class metrics:")
for cls in ["0", "1", "2", "3"]:
    if cls in cls_report:
        m = cls_report[cls]
        print(f"    Level {cls}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1-score']:.3f} (n={m['support']})")

print(f"\n  Macro-F1:  {cls_report['macro avg']['f1-score']:.4f}")
print(f"  Weighted-F1: {cls_report['weighted avg']['f1-score']:.4f}")

# ECE per class
print("\n  Calibration (ECE):")
for cls in range(4):
    ece = expected_calibration_error(
        (y_test == cls).astype(int).values, cal_probs[:, cls]
    )
    print(f"    Class {cls}: {ece:.4f}")

# Confusion matrix
print("\n  Confusion Matrix (rows=actual, cols=predicted):")
cm = report["confusion_matrix"]
print(f"  {'':>10} pred_0  pred_1  pred_2  pred_3")
for i, row in enumerate(cm):
    print(f"  actual_{i}  {row[0]:>6}  {row[1]:>6}  {row[2]:>6}  {row[3]:>6}")

# Save model
model_path = save_model(model, str(resolve_path("models")), "severity_lgbm")
print(f"\nModel saved: {model_path}")
print("\nPhase 2 complete!")
