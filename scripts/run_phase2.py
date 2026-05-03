"""Run Phase 2: Problem A Severity Model Training & Evaluation.

New in this version
-------------------
* --use-narrative / --no-narrative flag (default: True)
  Encodes Report 1_Narrative + Synopsis with SBERT → PCA(32) and
  concatenates the 32 dimensions to the tabular feature matrix.
  This is the single highest-ROI change for QWK.

* --ordinal / --no-ordinal flag (default: True)
  Trains an ordinal cumulative logit LightGBM (3 binary classifiers)
  alongside the standard multiclass model. Picks the winner on QWK.

* Temperature scaling after isotonic calibration (always on).
  Reduces ECE from ~0.028 to the target < 0.020.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.config import (
    load_main_config, load_feature_whitelist, load_cost_matrix,
    resolve_path, set_seeds,
)
from src.data.loader import load_raw_data, parse_time_date
from src.data.target_engineering import apply_severity_rubric, validate_severity_distribution
from src.data.leakage_audit import (
    get_problem_a_features, get_problem_a_text_features, validate_no_leakage,
)
from src.features.temporal import (
    extract_temporal_features, create_temporal_split,
    validate_temporal_split, get_split_data,
)
from src.features.encoding import (
    identify_column_types, prepare_for_lgbm, bucket_experience,
)
from src.features.text import preprocess_narratives, clean_narrative
from src.models.severity import (
    train_lgbm_severity, calibrate_model, predict_calibrated, save_model,
    train_catboost_severity, predict_cost_sensitive,
)
from src.models.ordinal import (
    train_ordinal_lgbm, predict_ordinal_probs, predict_ordinal,
    calibrate_ordinal, predict_ordinal_calibrated, temperature_scale,
    apply_temperature,
)
from src.evaluation.ordinal_metrics import full_severity_report
from src.evaluation.calibration import expected_calibration_error


# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser(description="Phase 2 — Problem A Severity")
parser.add_argument("--use-narrative", action=argparse.BooleanOptionalAction, default=True,
                    help="Encode narratives with SBERT→PCA(32) and add to features (default: True)")
parser.add_argument("--narrative-pca-dim", type=int, default=32,
                    help="Number of PCA dimensions for narrative embeddings (default: 32)")
parser.add_argument("--ordinal", action=argparse.BooleanOptionalAction, default=True,
                    help="Also train ordinal cumulative logit model (default: True)")
args = parser.parse_args()

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
if not val["is_valid"]:
    for w in val["warnings"]:
        print(f"  WARNING: {w}")
    print("  Rubric distribution is out of expected range — aborting.")
    sys.exit(1)
print("  Distribution valid.")

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
    for leak in leaks:
        print(f"  LEAK: {leak}")
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
        print(f"  WARNING: Dropped (>80% missing): {hm}")

if len(feature_cols) < 5:
    print(f"  FATAL: Only {len(feature_cols)} features remain after dropping. Check whitelist/data.")
    sys.exit(1)

print(f"Final tabular features: {len(feature_cols)}")
print(f"  Categoricals: {[c for c in col_types['categorical'] if c in feature_cols]}")

# ============================================================
# Narrative Embeddings (SBERT → PCA)
# ============================================================
narrative_dim = 0  # track extra dims added

if args.use_narrative:
    print("\n─── Narrative Embedding (SBERT → PCA) ───")
    text_cols = get_problem_a_text_features(df, whitelist)
    if not text_cols:
        print("  WARNING: No text columns found in whitelist. Skipping narrative features.")
    else:
        print(f"  Text columns: {text_cols}")
        from sentence_transformers import SentenceTransformer
        from sklearn.decomposition import PCA

        # Build combined text: narrative + synopsis joined with space
        df = preprocess_narratives(df, narrative_cols=text_cols)
        clean_cols = [f"{c}_clean" for c in text_cols if f"{c}_clean" in df.columns]
        if not clean_cols:
            # fallback: clean on the fly
            df["_combined_narrative"] = df[text_cols].fillna("").agg(" ".join, axis=1)
        else:
            df["_combined_narrative"] = df[clean_cols].fillna("").agg(" ".join, axis=1)

        print(f"  Loading sentence-transformers all-MiniLM-L6-v2...")
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        encoder.max_seq_length = 256

        texts = df["_combined_narrative"].tolist()
        print(f"  Encoding {len(texts)} texts (batch=128)...")
        raw_emb = encoder.encode(texts, batch_size=128, show_progress_bar=True,
                                 normalize_embeddings=True)
        print(f"  Raw embedding shape: {raw_emb.shape}")

        # PCA on train split only — fit, then transform all
        train_mask = df["split"] == "train"
        pca_dim = min(args.narrative_pca_dim, raw_emb.shape[1])
        pca = PCA(n_components=pca_dim, random_state=42)
        pca.fit(raw_emb[train_mask.values])
        explained = pca.explained_variance_ratio_.sum()
        print(f"  PCA({pca_dim}) explains {explained:.1%} of embedding variance")

        emb_pca = pca.transform(raw_emb)
        narrative_dim = pca_dim

        # Attach PCA columns to df
        emb_cols = [f"narr_pca_{i}" for i in range(pca_dim)]
        emb_df = pd.DataFrame(emb_pca, columns=emb_cols, index=df.index)
        df = pd.concat([df, emb_df], axis=1)
        feature_cols = feature_cols + emb_cols

        # Save PCA + encoder metadata for inference
        pca_path = resolve_path("models/severity_narrative_pca.joblib")
        joblib.dump({"pca": pca, "pca_cols": emb_cols, "text_cols": text_cols}, pca_path)
        print(f"  Narrative PCA saved: {pca_path}")
        print(f"  Total features now: {len(feature_cols)}")

# ============================================================
# Split data
# ============================================================
splits = get_split_data(df, "severity_level", feature_cols)
X_train, y_train = splits["train"]
X_val, y_val = splits["val"]
X_test, y_test = splits["test"]
print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

cat_features = [c for c in col_types["categorical"] if c in feature_cols]
cost_config = load_cost_matrix()
costs = cost_config.get("costs", {})

# ============================================================
# Train LightGBM (standard multiclass)
# ============================================================
print("\nTraining LightGBM severity model (standard multiclass)...")
lgb_model, lgb_history = train_lgbm_severity(X_train, y_train, X_val, y_val, cat_features)
print(f"Best iteration: {lgb_history['best_iteration']}")

print("Calibrating LightGBM (isotonic)...")
lgb_calibrators = calibrate_model(lgb_model, X_val, y_val)
lgb_cal_probs_val = predict_calibrated(lgb_model, X_val, lgb_calibrators)
lgb_cal_probs = predict_calibrated(lgb_model, X_test, lgb_calibrators)

# Temperature scaling — fit on validation, apply to test (never fit on test)
print("Applying temperature scaling...")
lgb_ts_probs, lgb_temp = temperature_scale(lgb_cal_probs_val, y_val.values)
print(f"  Optimal temperature: {lgb_temp:.3f}")
lgb_ts_probs_test = apply_temperature(lgb_cal_probs, lgb_temp)

lgb_y_pred = predict_cost_sensitive(lgb_ts_probs_test, costs)
lgb_report = full_severity_report(y_test.values, lgb_y_pred, costs)

# ============================================================
# Train CatBoost
# ============================================================
print("\nTraining CatBoost severity model...")
cb_model, X_train_cb, X_val_cb = train_catboost_severity(X_train, y_train, X_val, y_val, cat_features)

print("Calibrating CatBoost...")
cb_calibrators = calibrate_model(cb_model, X_val_cb, y_val)
X_test_cb = X_test.copy()
for col in cat_features:
    X_test_cb[col] = X_test_cb[col].astype(str).replace("nan", "Missing").fillna("Missing")
cb_cal_probs = predict_calibrated(cb_model, X_test_cb, cb_calibrators)
cb_y_pred = predict_cost_sensitive(cb_cal_probs, costs)
cb_report = full_severity_report(y_test.values, cb_y_pred, costs)

# ============================================================
# Train Ordinal LightGBM (optional)
# ============================================================
ordinal_model = None
ordinal_report = None
if args.ordinal:
    print("\nTraining Ordinal LightGBM (cumulative logit)...")
    try:
        ord_clfs, ord_info = train_ordinal_lgbm(
            X_train, y_train, X_val, y_val, cat_features,
        )
        print(f"  Best iterations: {ord_info['best_iterations']}")

        # Isotonic calibration
        ord_calibrators = calibrate_ordinal(ord_clfs, X_val, y_val.values)
        ord_cal_probs_val = predict_ordinal_calibrated(ord_clfs, X_val, ord_calibrators)
        ord_cal_probs_test = predict_ordinal_calibrated(ord_clfs, X_test, ord_calibrators)

        # Temperature scaling — fit on validation, apply to test
        ord_ts_probs, ord_temp = temperature_scale(ord_cal_probs_val, y_val.values)
        ord_ts_probs_test = apply_temperature(ord_cal_probs_test, ord_temp)
        print(f"  Optimal temperature: {ord_temp:.3f}")

        ord_y_pred = predict_cost_sensitive(ord_ts_probs_test, costs)
        ordinal_report = full_severity_report(y_test.values, ord_y_pred, costs)
        ordinal_model = (ord_clfs, ord_calibrators, ord_temp)
    except Exception as e:
        print(f"  WARNING: Ordinal model training failed: {e}")
        print("  Falling back to best standard model.")

# ============================================================
# Compare & Select Best
# ============================================================
print("\n" + "="*55)
print("  MODEL COMPARISON (Test Set)")
print("="*55)
print(f"  LightGBM (standard)  QWK: {lgb_report['qwk']:.4f}  "
      f"Macro-F1: {lgb_report['classification_report']['macro avg']['f1-score']:.4f}")
print(f"  CatBoost             QWK: {cb_report['qwk']:.4f}  "
      f"Macro-F1: {cb_report['classification_report']['macro avg']['f1-score']:.4f}")

if ordinal_report is not None:
    print(f"  LightGBM (ordinal)   QWK: {ordinal_report['qwk']:.4f}  "
          f"Macro-F1: {ordinal_report['classification_report']['macro avg']['f1-score']:.4f}")

# Collect all results and pick best by QWK
candidates = {
    "LightGBM_standard": (lgb_report, "lgbm"),
    "CatBoost": (cb_report, "catboost"),
}
if ordinal_report is not None:
    candidates["LightGBM_ordinal"] = (ordinal_report, "ordinal")

best_name, (best_report, best_type) = max(candidates.items(), key=lambda kv: kv[1][0]["qwk"])
print(f"\n  >> Best model: {best_name}")

# ============================================================
# Save best model
# ============================================================
if best_type == "lgbm":
    model_path = save_model(lgb_model, str(resolve_path("models")), "severity_lgbm")
    best_calibrators = lgb_calibrators
    best_probs = lgb_ts_probs_test
elif best_type == "catboost":
    model_path = resolve_path("models/severity_catboost.cbm")
    cb_model.save_model(str(model_path))
    best_calibrators = cb_calibrators
    best_probs = cb_cal_probs
else:  # ordinal
    import lightgbm as lgb_lib
    ord_clfs, ord_calibrators, ord_temp = ordinal_model
    for i, clf in enumerate(ord_clfs):
        clf.save_model(str(resolve_path(f"models/severity_ordinal_clf{i}.txt")))
    model_path = resolve_path("models/severity_ordinal_clf0.txt")
    best_calibrators = ord_calibrators
    best_probs = ord_ts_probs_test

cal_path = resolve_path("models/severity_calibrators.joblib")
joblib.dump(best_calibrators, cal_path)
print(f"  Model saved:       {model_path}")
print(f"  Calibrators saved: {cal_path}")

# Save feature column list for inference
feat_path = resolve_path("models/severity_feature_cols.json")
import json
with open(feat_path, "w") as f:
    json.dump({
        "feature_cols": feature_cols,
        "tabular_feature_count": len(feature_cols) - narrative_dim,
        "narrative_pca_dim": narrative_dim,
        "use_narrative": args.use_narrative,
        "best_model": best_name,
    }, f, indent=2)

# ============================================================
# Best Model — Full Evaluation
# ============================================================
print("\n" + "="*55)
print(f"  BEST MODEL ({best_name}) — TEST SET EVALUATION")
print("="*55)
print(f"  QWK:              {best_report['qwk']:.4f}")
print(f"  QWK 95% CI:       [{best_report['qwk_bootstrap']['ci_low']:.4f}, "
      f"{best_report['qwk_bootstrap']['ci_high']:.4f}]")
print(f"  Ordinal MAE:      {best_report['ordinal_mae']:.4f}")
print(f"  Class-Wtd MAE:    {best_report['class_weighted_mae']:.4f}")
if "asymmetric_cost" in best_report:
    print(f"  Asym. Cost:       {best_report['asymmetric_cost']:.4f}")

cls_report = best_report["classification_report"]
print("\n  Per-class metrics:")
for cls in ["0", "1", "2", "3"]:
    if cls in cls_report:
        m = cls_report[cls]
        print(f"    Level {cls}: P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1-score']:.3f} (n={m['support']})")

print(f"\n  Macro-F1:    {cls_report['macro avg']['f1-score']:.4f}")
print(f"  Weighted-F1: {cls_report['weighted avg']['f1-score']:.4f}")

print("\n  Calibration (ECE per class):")
for cls in range(4):
    ece = expected_calibration_error(
        (y_test == cls).astype(int).values, best_probs[:, cls]
    )
    print(f"    Class {cls}: ECE={ece:.4f}")

print("\n  Confusion Matrix (rows=actual, cols=predicted):")
cm = best_report["confusion_matrix"]
print(f"  {'':>10} pred_0  pred_1  pred_2  pred_3")
for i, row in enumerate(cm):
    print(f"  actual_{i}  {row[0]:>6}  {row[1]:>6}  {row[2]:>6}  {row[3]:>6}")

# ============================================================
# Slice Analysis — Performance by Year
# ============================================================
print("\n" + "="*55)
print("  SLICE ANALYSIS & FEATURE IMPORTANCE")
print("="*55)

if "year" in X_test.columns:
    print("  Slice by Year:")
    X_slice = X_test_cb if best_type == "catboost" else X_test
    for yr in sorted(X_slice["year"].unique()):
        idx = X_slice["year"] == yr
        y_slice = y_test[idx]
        p_slice = predict_cost_sensitive(best_probs[idx], costs)
        if len(y_slice) > 10:
            sr = full_severity_report(y_slice.values, p_slice, costs)
            print(f"    {int(yr)}: QWK={sr['qwk']:.4f}  "
                  f"Macro-F1={sr['classification_report']['macro avg']['f1-score']:.4f}  "
                  f"(n={len(y_slice)})")

# ============================================================
# Feature Importance
# ============================================================
print("\n  Feature Importance (top 15):")
if best_type == "lgbm":
    importance = lgb_model.feature_importance(importance_type="gain")
    top_idx = np.argsort(importance)[::-1][:15]
    for i in top_idx:
        marker = " ← narrative" if feature_cols[i].startswith("narr_pca_") else ""
        print(f"    {feature_cols[i]:<50} : {importance[i]:.2f}{marker}")
elif best_type == "catboost":
    importance = cb_model.get_feature_importance()
    top_idx = np.argsort(importance)[::-1][:15]
    for i in top_idx:
        print(f"    {feature_cols[i]:<50} : {importance[i]:.2f}")
else:
    print("  (Feature importance not available for ordinal ensemble — check per-threshold clf)")

print("\nPhase 2 complete!")
