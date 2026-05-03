"""
Problem C — Preflight Risk Model Evaluation
============================================
Standalone offline evaluation script.  Runs without re-training:
  1. Loads preflight_features_final.parquet + preflight_lgbm_calibrated.joblib
  2. Creates a temporal test split (last 20% of data by date)
  3. Applies PriorShiftedCalibratedModel to get calibrated P(incident)
  4. Computes ROC-AUC, PR-AUC, Brier Score, ECE at the heuristic threshold
  5. Runs threshold tuning sweep (F1-maximising)
  6. Prints a full report and saves to models/preflight_eval_results.json

Usage
-----
    python scripts/evaluate_preflight.py
    python scripts/evaluate_preflight.py --optimize recall   # favour sensitivity
    python scripts/evaluate_preflight.py --test-frac 0.25   # larger test set
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import resolve_path
from src.models.preflight import PriorShiftedCalibratedModel
from src.models.preflight_improvements import (
    tune_preflight_threshold,
    evaluate_preflight_model,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate Problem C preflight risk model")
parser.add_argument(
    "--features-path", default=None,
    help="Path to preflight_features_final.parquet. Default: auto-discover."
)
parser.add_argument(
    "--model-path", default=None,
    help="Path to preflight_lgbm_calibrated.joblib. Default: auto-discover."
)
parser.add_argument(
    "--test-frac", type=float, default=0.20,
    help="Fraction of data to use as test set (latest records). Default: 0.20"
)
parser.add_argument(
    "--optimize", choices=["f1", "precision", "recall", "j_statistic"], default="f1",
    help="Metric to maximise during threshold sweep. Default: f1"
)
parser.add_argument(
    "--true-prior", type=float, default=None,
    help="True incident base rate. If None, reads from model artifact."
)
args = parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Discover artefact paths
# ─────────────────────────────────────────────────────────────────────────────
def _find(candidates):
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    return None


features_path = args.features_path or _find([
    resolve_path("data/processed/preflight_features_final.parquet"),
    resolve_path("data/processed/preflight_features.parquet"),
])
model_path = args.model_path or _find([
    resolve_path("models/preflight_lgbm_calibrated.joblib"),
])

if features_path is None:
    print("ERROR: Could not find preflight_features_final.parquet.")
    print("  Run notebooks/05e_preflight_features.ipynb first.")
    sys.exit(1)
if model_path is None:
    print("ERROR: Could not find preflight_lgbm_calibrated.joblib.")
    print("  Run notebooks/05f_preflight_model.ipynb first.")
    sys.exit(1)

print(f"Features : {features_path}")
print(f"Model    : {model_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading features...")
df = pd.read_parquet(features_path)
print(f"  Shape: {df.shape}")

# Detect label column
label_candidates = ["incident", "label", "is_incident", "y", "target"]
label_col = next((c for c in label_candidates if c in df.columns), None)
if label_col is None:
    print(f"ERROR: Could not find label column. Available: {df.columns.tolist()[:20]}")
    sys.exit(1)
print(f"  Label column: '{label_col}'  | Positive rate: {df[label_col].mean():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Temporal test split
# ─────────────────────────────────────────────────────────────────────────────
# Sort by date column if available; otherwise by index
date_candidates = ["fl_date", "date", "Date", "incident_date", "flight_date"]
date_col = next((c for c in date_candidates if c in df.columns), None)

if date_col is not None:
    df = df.sort_values(date_col).reset_index(drop=True)
    print(f"  Sorted by '{date_col}': {df[date_col].min()} → {df[date_col].max()}")
else:
    print("  WARNING: No date column found — using row order for temporal split.")

n_test = max(int(len(df) * args.test_frac), 50)
test_df = df.iloc[-n_test:].copy()
train_df = df.iloc[:-n_test].copy()

print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")
print(f"  Test positive rate: {test_df[label_col].mean():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading model...")
import joblib
artifact = joblib.load(model_path)

# Handle dictionary artifact or raw model
if isinstance(artifact, dict):
    model_obj = artifact.get("model")
    feature_cols_from_artifact = artifact.get("features", [])
    # Also look for priors in dict if not in model attributes
    true_prior_from_artifact = artifact.get("true_prior")
    train_prior_from_artifact = artifact.get("train_prior")
else:
    model_obj = artifact
    feature_cols_from_artifact = []
    true_prior_from_artifact = None
    train_prior_from_artifact = None

# Prior detection
true_prior = args.true_prior
if true_prior is None:
    if hasattr(model_obj, "true_prior"):
        true_prior = model_obj.true_prior
    elif true_prior_from_artifact is not None:
        true_prior = true_prior_from_artifact
    else:
        # Estimate from full data positive rate
        true_prior = float(df[label_col].mean())
        print(f"  True prior (estimated from data): {true_prior:.4f}")

train_prior = None
if hasattr(model_obj, "train_prior"):
    train_prior = model_obj.train_prior
elif train_prior_from_artifact is not None:
    train_prior = train_prior_from_artifact
else:
    # Most preflight models use 0.5 (balanced case-control)
    train_prior = 0.5

if true_prior == 0:
    true_prior = 0.05
    print(f"  WARNING: Adjusted true_prior to 0.05 (was 0)")

print(f"  True prior:  {true_prior:.4f}")
print(f"  Train prior: {train_prior:.4f}")

# Determine feature columns
if feature_cols_from_artifact:
    feature_cols = feature_cols_from_artifact
elif hasattr(model_obj, "features"):
    feature_cols = model_obj.features
else:
    # Exclude non-feature columns
    exclude = {label_col, date_col, "split", "acn_num_ACN", "index"}
    feature_cols = [c for c in df.columns if c not in exclude and c is not None]

feature_cols = [c for c in feature_cols if c in test_df.columns]
print(f"  Feature columns used: {len(feature_cols)}")

# Wrap in PriorShiftedCalibratedModel for correct evaluation
model = PriorShiftedCalibratedModel(
    calibrated_model=model_obj,
    true_prior=true_prior,
    train_prior=train_prior,
    features=feature_cols
)

heuristic_threshold = true_prior * 5.0
print(f"  Heuristic threshold (true_prior × 5): {heuristic_threshold:.4f}")

X_test = test_df[feature_cols].copy()
y_test = test_df[label_col].values.astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating predictions...")
try:
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test)
    print(f"  Probability range: [{y_prob.min():.4f}, {y_prob.max():.4f}]")
    print(f"  Mean predicted P(incident): {y_prob.mean():.4f}")
except Exception as e:
    print(f"ERROR during prediction: {e}")
    print("  Trying with features subset from model.features...")
    raise


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate at heuristic threshold
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  EVALUATION AT HEURISTIC THRESHOLD")
print("="*55)
heuristic_results = evaluate_preflight_model(y_test, y_prob, threshold=heuristic_threshold)
for k, v in heuristic_results.items():
    if isinstance(v, float):
        print(f"  {k:<25}: {v:.4f}")
    elif isinstance(v, dict) and k == "confusion_matrix":
        cm = v
        print(f"  Confusion matrix:")
        print(f"    TN={cm['tn']:>6}  FP={cm['fp']:>6}")
        print(f"    FN={cm['fn']:>6}  TP={cm['tp']:>6}")
    elif not isinstance(v, dict):
        print(f"  {k:<25}: {v}")


# ─────────────────────────────────────────────────────────────────────────────
# Threshold sweep
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print(f"  THRESHOLD SWEEP (maximising {args.optimize})")
print("="*55)
tune_result = tune_preflight_threshold(
    y_test, y_prob,
    true_prior=true_prior,
    multiplier_range=(1.0, 20.0),
    n_steps=200,
    optimize=args.optimize,
)
print(f"  Heuristic threshold:  {heuristic_threshold:.5f} (×{5.0:.1f})")
print(f"  Optimal threshold:    {tune_result['best_threshold']:.5f} "
      f"(×{tune_result['best_multiplier']:.2f})")
print(f"  Best {args.optimize}:          {tune_result['best_score']:.4f}")

# Evaluate at tuned threshold
print("\n" + "="*55)
print("  EVALUATION AT TUNED THRESHOLD")
print("="*55)
tuned_results = evaluate_preflight_model(y_test, y_prob, threshold=tune_result["best_threshold"])
for k, v in tuned_results.items():
    if isinstance(v, float):
        print(f"  {k:<25}: {v:.4f}")
    elif isinstance(v, dict) and k == "confusion_matrix":
        cm = v
        print(f"  Confusion matrix:")
        print(f"    TN={cm['tn']:>6}  FP={cm['fp']:>6}")
        print(f"    FN={cm['fn']:>6}  TP={cm['tp']:>6}")

# Quality gate check
roc = tuned_results["roc_auc"]
pr = tuned_results["pr_auc"]
print("\n  Quality Gates:")
print(f"    ROC-AUC >= 0.72: {'PASSED' if roc >= 0.72 else f'FAILED ({roc:.4f})'}")
print(f"    PR-AUC  >= 0.12: {'PASSED' if pr >= 0.12 else f'FAILED ({pr:.4f})'}")


# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────
output = {
    "features_path": str(features_path),
    "model_path": str(model_path),
    "n_train": len(train_df),
    "n_test": len(test_df),
    "test_positive_rate": float(y_test.mean()),
    "true_prior": float(true_prior),
    "heuristic_threshold": float(heuristic_threshold),
    "heuristic_results": {k: v for k, v in heuristic_results.items() if isinstance(v, (float, int))},
    "threshold_tuning": {
        "optimize": args.optimize,
        "best_threshold": tune_result["best_threshold"],
        "best_multiplier": tune_result["best_multiplier"],
        "best_score": tune_result["best_score"],
    },
    "tuned_results": {k: v for k, v in tuned_results.items() if isinstance(v, (float, int))},
}

out_path = resolve_path("models/preflight_eval_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved: {out_path}")
print("\nPhase C evaluation complete!")
