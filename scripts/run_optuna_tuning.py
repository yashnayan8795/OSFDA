"""Run Optuna hyperparameter tuning for the severity model.

Constraints enforced
--------------------
* Narratives scanned for rubric-leaking phrases before SBERT encoding.
* PCA(128) is fit ONCE on the training split before any Optuna trial.
  Within each trial, dimensions are dynamically sliced (narr_pca_0..N-1)
  so PCA is never refit per trial.
* --narrative-pca-dim sweep: if set to 0, Optuna will suggest the best
  PCA dim from [32, 64, 96, 128] as a hyperparameter per trial.
* All tuning uses train + val only. Test set is never read here.
* Random seed 42 for all ops.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import pandas as pd
import numpy as np
import optuna
import joblib
import lightgbm as lgb

from src.utils.config import (
    load_main_config, load_feature_whitelist, load_cost_matrix,
    resolve_path, set_seeds,
)
from src.data.loader import load_raw_data, parse_time_date
from src.data.target_engineering import apply_severity_rubric
from src.data.rubric_scanner import prepare_safe_narratives, print_scan_report
from src.data.leakage_audit import get_problem_a_features, get_problem_a_text_features
from src.features.temporal import (
    extract_temporal_features, create_temporal_split, get_split_data,
)
from src.features.encoding import (
    identify_column_types, prepare_for_lgbm, bucket_experience,
)
from src.features.text import preprocess_narratives
from src.models.severity import (
    build_lgbm_dataset, train_lgbm_severity, calibrate_model,
    predict_calibrated, predict_cost_sensitive, save_model,
)
from src.models.ordinal import (
    train_ordinal_lgbm, predict_ordinal, ordinal_optuna_objective,
    temperature_scale, apply_temperature,
)
from src.evaluation.ordinal_metrics import full_severity_report, quadratic_weighted_kappa
from src.evaluation.calibration import expected_calibration_error

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Optuna tuning for Problem A severity")
parser.add_argument("--n-trials", type=int, default=200,
                    help="Number of Optuna trials (default: 200)")
parser.add_argument("--timeout", type=int, default=3600,
                    help="Max seconds to run Optuna (default: 3600)")
parser.add_argument("--use-narrative", action=argparse.BooleanOptionalAction, default=True,
                    help="Include SBERT→PCA narrative features (default: True)")
parser.add_argument("--narrative-pca-dim", type=int, default=32,
                    help="PCA dims: 32, 64, 96, 128. Set 0 to let Optuna sweep [32,64,96,128] (default: 32)")
parser.add_argument("--ordinal", action=argparse.BooleanOptionalAction, default=False,
                    help="Use ordinal cumulative logit objective (default: False)")
args = parser.parse_args()

set_seeds()
config = load_main_config()

# ─────────────────────────────────────────────────────────────────────────────
# Data loading & preparation (identical to run_phase2.py)
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# Narrative embeddings (Phase 0: rubric scan, then PCA(128) fit ONCE)
# If --narrative-pca-dim=0, Optuna will sweep [32,64,96,128] per trial.
# ─────────────────────────────────────────────────────────────────────────────
narrative_dim = 0
PCA_DIM_SWEEP_MAX = 128       # Always fit PCA at max depth; slice per trial
PCA_DIM_SWEEP_OPTIONS = [32, 64, 96, 128]
raw_emb = None
_pca128 = None                # Pre-fitted PCA(128) — fit once, used for all trials
_emb_pca128 = None            # All samples embedded at 128 dims

if args.use_narrative:
    print("\n─── Narrative Safety Scan (Phase 0) ───")
    text_cols = get_problem_a_text_features(df, whitelist)
    if text_cols:
        # 1. Scan and mask rubric-leaking phrases
        df, scan_report = prepare_safe_narratives(df, text_cols=text_cols, mask=True)
        print_scan_report(scan_report)

        safe_text_cols = [f"{c}_safe" for c in text_cols if f"{c}_safe" in df.columns]
        combined_col = "_combined_narrative_safe"
        df[combined_col] = df[safe_text_cols].fillna("").agg(" ".join, axis=1)

        # 2. Encode with SBERT
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        encoder.max_seq_length = 256
        print(f"  Encoding {len(df)} texts (batch=128)...")
        raw_emb = encoder.encode(df[combined_col].tolist(), batch_size=128,
                                 show_progress_bar=True, normalize_embeddings=True)
        print(f"  Raw embedding shape: {raw_emb.shape}")

        # 3. Fit PCA(128) ONCE on training split only
        from sklearn.decomposition import PCA
        train_mask = df["split"] == "train"
        _pca128 = PCA(n_components=PCA_DIM_SWEEP_MAX, random_state=42)
        _pca128.fit(raw_emb[train_mask.values])
        _emb_pca128 = _pca128.transform(raw_emb)  # shape: (N, 128)

        explained = _pca128.explained_variance_ratio_.cumsum()
        print(f"  PCA(128) cumulative variance: "
              f"@32={explained[31]:.1%}  @64={explained[63]:.1%}  "
              f"@96={explained[95]:.1%}  @128={explained[127]:.1%}")

        # If pca_dim is fixed (not sweep mode), attach those dims to df now
        if args.narrative_pca_dim > 0:
            pca_dim = min(args.narrative_pca_dim, PCA_DIM_SWEEP_MAX)
            emb_cols = [f"narr_pca_{i}" for i in range(pca_dim)]
            emb_df = pd.DataFrame(_emb_pca128[:, :pca_dim], columns=emb_cols, index=df.index)
            df = pd.concat([df, emb_df], axis=1)
            feature_cols = feature_cols + emb_cols
            narrative_dim = pca_dim
            print(f"  PCA({pca_dim}) attached. Total features: {len(feature_cols)}")
        else:
            # Sweep mode: attach all 128 dims; Optuna will slice per trial
            emb_cols = [f"narr_pca_{i}" for i in range(PCA_DIM_SWEEP_MAX)]
            emb_df = pd.DataFrame(_emb_pca128, columns=emb_cols, index=df.index)
            df = pd.concat([df, emb_df], axis=1)
            # feature_cols is intentionally NOT updated here;
            # each trial will add narr_pca_0..{dim-1} dynamically
            narrative_dim = PCA_DIM_SWEEP_MAX
            print(f"  PCA(128) sweep mode: Optuna will select best dim from {PCA_DIM_SWEEP_OPTIONS}")
    else:
        print("  WARNING: No text columns found. Skipping narrative features.")


# ─────────────────────────────────────────────────────────────────────────────
# Split
# ─────────────────────────────────────────────────────────────────────────────
splits = get_split_data(df, "severity_level", feature_cols)
X_train, y_train = splits["train"]
X_val, y_val = splits["val"]
X_test, y_test = splits["test"]
cat_features = [c for c in col_types["categorical"] if c in feature_cols]

print(f"Features: {len(feature_cols)}, Train: {len(X_train)}, Val: {len(X_val)}")

# ─────────────────────────────────────────────────────────────────────────────
# Optuna study
# ─────────────────────────────────────────────────────────────────────────────
study_name = f"severity_{'ordinal' if args.ordinal else 'lgbm'}_narrative{narrative_dim}"

if args.ordinal:
    def objective(trial):
        return ordinal_optuna_objective(
            trial, X_train, y_train.values, X_val, y_val.values, cat_features
        )
else:
    def objective(trial):
        # ── Dynamic PCA dim slicing (sweep mode only) ────────────────────────
        # In fixed mode (args.narrative_pca_dim > 0), feature_cols already
        # contains the correct narr_pca_* columns.
        # In sweep mode (args.narrative_pca_dim == 0), we slice on-the-fly.
        _X_train = X_train
        _X_val = X_val
        _cat_features = cat_features

        if args.narrative_pca_dim == 0 and _emb_pca128 is not None:
            trial_pca_dim = trial.suggest_categorical("pca_dim", PCA_DIM_SWEEP_OPTIONS)
            # Build augmented train/val with sliced PCA cols (NOT modifying global df)
            _narr_cols = [f"narr_pca_{i}" for i in range(trial_pca_dim)]
            # Create augmented feature sets for this trial only
            _base_cols = [c for c in feature_cols if not c.startswith("narr_pca_")]
            _X_train = X_train[_base_cols].copy()
            _X_val = X_val[_base_cols].copy()
            splits_trial = get_split_data(df, "severity_level", _base_cols + _narr_cols)
            _X_train, _ = splits_trial["train"]
            _X_val, _ = splits_trial["val"]
            _cat_features = [c for c in cat_features if c in _X_train.columns]

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

        dtrain, dval = build_lgbm_dataset(_X_train, y_train, _X_val, y_val, _cat_features)
        callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
        model = lgb.train(
            params, dtrain, num_boost_round=2000,
            valid_sets=[dval], valid_names=["val"],
            callbacks=callbacks,
        )

        val_probs = model.predict(_X_val)
        val_pred = val_probs.argmax(axis=1)
        return quadratic_weighted_kappa(y_val, val_pred)  # Maximize QWK


storage_path = resolve_path("data/processed/optuna.db")
storage_path.parent.mkdir(parents=True, exist_ok=True)
storage_url = f"sqlite:///{storage_path}"
study = optuna.create_study(
    direction="maximize",
    study_name=study_name,
    storage=storage_url,
    load_if_exists=True,
)
study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=True)

print(f"\nBest QWK (val): {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# ─────────────────────────────────────────────────────────────────────────────
# Retrain with best params on full train+val
# ─────────────────────────────────────────────────────────────────────────────
print("\nRetraining with best params on train+val combined...")
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
cal_probs_val = predict_calibrated(model, X_val, calibrators)
cal_probs_test = predict_calibrated(model, X_test, calibrators)

# Fit temperature on validation, apply to test — never fit on test
_, best_temp = temperature_scale(cal_probs_val, y_val.values)
print(f"Optimal temperature: {best_temp:.3f}")
ts_probs_test = apply_temperature(cal_probs_test, best_temp)

cost_config = load_cost_matrix()
y_pred = predict_cost_sensitive(ts_probs_test, cost_config.get("costs"))
report = full_severity_report(y_test.values, y_pred, cost_config.get("costs"))

print("\n" + "="*55)
print("  TUNED MODEL — TEST SET EVALUATION")
print("="*55)
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
joblib.dump(calibrators, resolve_path("models/severity_calibrators_tuned.joblib"))
print(f"\nTuned model saved: {model_path}")

# Save study summary
summary_path = resolve_path("models/optuna_study_summary.json")
with open(summary_path, "w") as f:
    json.dump({
        "study_name": study_name,
        "n_trials_completed": len(study.trials),
        "best_qwk": study.best_value,
        "best_params": study.best_params,
        "narrative_pca_dim": narrative_dim,
        "use_narrative": args.use_narrative,
    }, f, indent=2)
print(f"Optuna summary: {summary_path}")
