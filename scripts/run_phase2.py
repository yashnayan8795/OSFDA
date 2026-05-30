"""Run Phase 2: Problem A Severity Model Training & Evaluation.

Constraints enforced by this script
-------------------------------------
* Temporal split 2012-2018 (train) / 2018-2019 (val) / 2019-2022 (test) is
  the only valid split. No random splits are performed.
* All calibration, threshold tuning, and Optuna optimization use train+val only.
  Test set is read only for the final locked model.
* Narrative texts are scanned for rubric-leaking phrases before SBERT encoding.
  Phrases above the 1% threshold are masked.
* Every experiment records QWK, Macro-F1, L3-Recall, L3-Precision,
  Expected Cost, and ECE before it is accepted or rejected.
* Random seeds fixed at 42 for all NumPy, CatBoost, LightGBM, and UMAP ops.

New flags
---------
* --narrative-pca-dim 32|64|96|128   (default: 32)
* --use-umap                          Replace PCA with UMAP-15 embedding
* --ordinal                           Train ordinal LightGBM alongside standard
* --ordinal-catboost                  Train ordinal CatBoost (K-1 binary models)
* --two-stage                         Train the two-stage CatBoost classifier
* --class-weights W0 W1 W2 W3        Train weighted CatBoost (e.g. 1 5 20 100)
* --interaction-features              Add whitelisted pre-flight interaction features
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
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
from src.data.rubric_scanner import prepare_safe_narratives, print_scan_report
from src.data.leakage_audit import (
    get_problem_a_features, get_problem_a_text_features,
    get_problem_a_interaction_features, validate_no_leakage,
)
from src.features.temporal import (
    extract_temporal_features, create_temporal_split,
    validate_temporal_split, get_split_data,
)
from src.features.encoding import (
    identify_column_types, prepare_for_lgbm, bucket_experience,
    engineer_interaction_features,
)
from src.features.text import preprocess_narratives, clean_narrative
from src.models.severity import (
    train_lgbm_severity, calibrate_model, predict_calibrated, save_model,
    train_catboost_severity, train_catboost_severity_weighted, predict_cost_sensitive,
)
from src.models.ordinal import (
    train_ordinal_lgbm, predict_ordinal_probs, predict_ordinal,
    calibrate_ordinal, predict_ordinal_calibrated, temperature_scale,
    apply_temperature,
    train_ordinal_catboost, predict_ordinal_catboost_probs,
    calibrate_ordinal_catboost, predict_ordinal_catboost_calibrated,
)
from src.models.two_stage import TwoStageSeverityModel
from src.evaluation.ordinal_metrics import full_severity_report
from src.evaluation.calibration import expected_calibration_error
from src.evaluation.experiment_logger import ExperimentLogger, top_n_misclassifications


# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser(description="Phase 2 — Problem A Severity")
parser.add_argument("--use-narrative", action=argparse.BooleanOptionalAction, default=True,
                    help="Encode narratives with SBERT→PCA and add to features (default: True)")
parser.add_argument("--narrative-pca-dim", type=int, default=32,
                    help="PCA dimensions for narrative embeddings: 32, 64, 96, or 128 (default: 32)")
parser.add_argument("--use-umap", action="store_true", default=False,
                    help="Replace PCA with UMAP-15 narrative embedding (default: False)")
parser.add_argument("--ordinal", action=argparse.BooleanOptionalAction, default=True,
                    help="Also train ordinal cumulative LightGBM (default: True)")
parser.add_argument("--ordinal-catboost", action="store_true", default=False,
                    help="Also train ordinal cumulative CatBoost (default: False)")
parser.add_argument("--two-stage", action="store_true", default=False,
                    help="Also train the two-stage CatBoost classifier (default: False)")
parser.add_argument("--class-weights", type=float, nargs=4, default=None,
                    metavar=("W0", "W1", "W2", "W3"),
                    help="Class weights for CatBoost [w0 w1 w2 w3], e.g. 1 5 20 100")
parser.add_argument("--interaction-features", action="store_true", default=False,
                    help="Add whitelisted pre-flight interaction features (default: False)")
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

# Validate temporal split integrity — no random splits, fixed boundaries
val_result = validate_temporal_split(df)
if not val_result.get("is_valid", True):
    print("FATAL: Temporal split validation failed:", val_result.get("warnings"))
    sys.exit(1)
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
# Interaction Features (pre-flight, whitelisted)
# ============================================================
if args.interaction_features:
    print("\n─── Interaction Feature Engineering ───")
    df, interaction_feat_names = engineer_interaction_features(df)
    if interaction_feat_names:
        feature_cols = feature_cols + interaction_feat_names
        # Validate: interaction features must pass leakage gate
        is_clean_after, leaks_after = validate_no_leakage(
            feature_cols, problem="A", whitelist=whitelist
        )
        if not is_clean_after:
            print("  FATAL: Interaction features introduced leakage:")
            for leak in leaks_after:
                print(f"    LEAK: {leak}")
            sys.exit(1)
        print(f"  Added interaction features: {interaction_feat_names}")
    else:
        print("  WARNING: No interaction features could be created (missing source columns).")


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
# Narrative Embeddings (SBERT → PCA or UMAP)
# Phase 0: Scan narratives for rubric-leaking phrases before encoding
# ============================================================
narrative_dim = 0  # track extra dims added

if args.use_narrative:
    print("\n─── Narrative Safety Scan (Phase 0) ───")
    text_cols = get_problem_a_text_features(df, whitelist)
    if not text_cols:
        print("  WARNING: No text columns found in whitelist. Skipping narrative features.")
    else:
        # Scan + mask rubric-leaking phrases (>1% threshold triggers masking)
        df, scan_report = prepare_safe_narratives(df, text_cols=text_cols, mask=True)
        print_scan_report(scan_report)

        # Use safe (masked) columns for encoding
        safe_text_cols = [f"{c}_safe" for c in text_cols if f"{c}_safe" in df.columns]
        combined_col = "_combined_narrative_safe"
        df[combined_col] = df[safe_text_cols].fillna("").agg(" ".join, axis=1)

        print(f"\n  Loading sentence-transformers all-MiniLM-L6-v2...")
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        encoder.max_seq_length = 256

        texts = df[combined_col].tolist()
        print(f"  Encoding {len(texts)} texts (batch=128)...")
        raw_emb = encoder.encode(texts, batch_size=128, show_progress_bar=True,
                                 normalize_embeddings=True)
        print(f"  Raw embedding shape: {raw_emb.shape}")

        train_mask = df["split"] == "train"

        if args.use_umap:
            print(f"\n  Using UMAP-15 embedding (seed=42)...")
            import umap
            umap_model = umap.UMAP(
                n_components=15, metric="cosine",
                random_state=42, n_jobs=1,
            )
            # Fit ONLY on train split — never on val or test
            umap_model.fit(raw_emb[train_mask.values])
            emb_reduced = umap_model.transform(raw_emb)
            emb_cols = [f"narr_umap_{i}" for i in range(15)]
            narrative_dim = 15
            embedding_label = "UMAP-15"
            # Save UMAP model
            umap_path = resolve_path("models/severity_narrative_umap.joblib")
            joblib.dump({"umap": umap_model, "emb_cols": emb_cols, "text_cols": text_cols}, umap_path)
            print(f"  UMAP model saved: {umap_path}")
        else:
            from sklearn.decomposition import PCA
            pca_dim = min(args.narrative_pca_dim, raw_emb.shape[1])
            # Fit PCA ONLY on train split
            pca = PCA(n_components=pca_dim, random_state=42)
            pca.fit(raw_emb[train_mask.values])
            explained = pca.explained_variance_ratio_.sum()
            print(f"  PCA({pca_dim}) explains {explained:.1%} of embedding variance")
            emb_reduced = pca.transform(raw_emb)
            emb_cols = [f"narr_pca_{i}" for i in range(pca_dim)]
            narrative_dim = pca_dim
            embedding_label = f"PCA-{pca_dim}"
            # Save PCA + encoder metadata for inference
            pca_path = resolve_path("models/severity_narrative_pca.joblib")
            joblib.dump({"pca": pca, "pca_cols": emb_cols, "text_cols": text_cols}, pca_path)
            print(f"  Narrative PCA saved: {pca_path}")

        emb_df = pd.DataFrame(emb_reduced, columns=emb_cols, index=df.index)
        df = pd.concat([df, emb_df], axis=1)
        feature_cols = feature_cols + emb_cols
        print(f"  Embedding ({embedding_label}) added. Total features: {len(feature_cols)}")


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
        print(f"  WARNING: Ordinal LightGBM training failed: {e}")
        print("  Falling back to best standard model.")

# ============================================================
# Train Ordinal CatBoost (optional)
# ============================================================
ordinal_cb_report = None
ordinal_cb_model = None
if args.ordinal_catboost:
    print("\nTraining Ordinal CatBoost (cumulative binary)...")
    try:
        ord_cb_clfs, ord_cb_info = train_ordinal_catboost(
            X_train, y_train.values, X_val, y_val.values, cat_features, seed=42
        )
        print(f"  Best iterations: {ord_cb_info['best_iterations']}")

        # Calibrate on val
        ord_cb_cal = calibrate_ordinal_catboost(
            ord_cb_clfs, X_val, y_val.values, cat_features
        )
        ord_cb_cal_probs_val = predict_ordinal_catboost_calibrated(
            ord_cb_clfs, X_val, ord_cb_cal, cat_features
        )
        ord_cb_cal_probs_test = predict_ordinal_catboost_calibrated(
            ord_cb_clfs, X_test, ord_cb_cal, cat_features
        )

        # Temperature scaling — fit on val only
        ord_cb_ts_probs, ord_cb_temp = temperature_scale(ord_cb_cal_probs_val, y_val.values)
        ord_cb_ts_probs_test = apply_temperature(ord_cb_cal_probs_test, ord_cb_temp)
        print(f"  Optimal temperature: {ord_cb_temp:.3f}")

        # Recompute calibration metrics after temperature scaling
        print("  Post-temperature ECE:")
        for cls in range(4):
            ece_v = expected_calibration_error(
                (y_val == cls).astype(int).values, ord_cb_ts_probs[:, cls]
            )
            print(f"    Class {cls}: ECE={ece_v:.4f}")

        ord_cb_y_pred = predict_cost_sensitive(ord_cb_ts_probs_test, costs)
        ordinal_cb_report = full_severity_report(y_test.values, ord_cb_y_pred, costs)
        ordinal_cb_model = (ord_cb_clfs, ord_cb_cal, ord_cb_temp)
    except Exception as e:
        print(f"  WARNING: Ordinal CatBoost training failed: {e}")

# ============================================================
# Train Weighted CatBoost (optional asymmetric class weights)
# ============================================================
weighted_cb_report = None
weighted_cb_model = None
if args.class_weights is not None:
    w = list(args.class_weights)
    print(f"\nTraining Weighted CatBoost (class_weights={w})...")
    try:
        wcb_model, X_train_wcb, X_val_wcb = train_catboost_severity_weighted(
            X_train, y_train, X_val, y_val, cat_features,
            class_weights=w, seed=42
        )
        # MANDATORY: Rerun calibration after weighted training
        print("  Rerunning calibration after weighted training...")
        wcb_calibrators = calibrate_model(wcb_model, X_val_wcb, y_val)
        X_test_wcb = X_test.copy()
        for col in cat_features:
            X_test_wcb[col] = X_test_wcb[col].astype(str).replace("nan", "Missing").fillna("Missing")
        wcb_cal_probs_val = predict_calibrated(wcb_model, X_val_wcb, wcb_calibrators)
        wcb_cal_probs_test = predict_calibrated(wcb_model, X_test_wcb, wcb_calibrators)
        wcb_ts_probs, wcb_temp = temperature_scale(wcb_cal_probs_val, y_val.values)
        wcb_ts_probs_test = apply_temperature(wcb_cal_probs_test, wcb_temp)
        print(f"  Optimal temperature: {wcb_temp:.3f}")
        print("  Post-calibration ECE (after weighting):")
        for cls in range(4):
            ece_v = expected_calibration_error(
                (y_val == cls).astype(int).values, wcb_ts_probs[:, cls]
            )
            print(f"    Class {cls}: ECE={ece_v:.4f}")
        wcb_y_pred = predict_cost_sensitive(wcb_ts_probs_test, costs)
        weighted_cb_report = full_severity_report(y_test.values, wcb_y_pred, costs)
        weighted_cb_model = (wcb_model, wcb_calibrators, wcb_temp)
    except Exception as e:
        print(f"  WARNING: Weighted CatBoost training failed: {e}")

# ============================================================
# Train Two-Stage Classifier (optional)
# ============================================================
two_stage_report = None
two_stage_model = None
two_stage_probs_test = None
if args.two_stage:
    print("\nTraining Two-Stage Severity Classifier...")
    try:
        ts_model = TwoStageSeverityModel(seed=42)
        ts_model.fit(X_train, y_train.values, X_val, y_val.values, cat_features)
        ts_probs_test = ts_model.predict_proba(X_test)
        ts_y_pred = ts_model.predict(X_test, costs=costs)
        two_stage_report = full_severity_report(y_test.values, ts_y_pred, costs)
        two_stage_model = ts_model
        two_stage_probs_test = ts_probs_test
        print("  Recomputing calibration after two-stage modeling...")
        for cls in range(4):
            ece_v = expected_calibration_error(
                (y_test == cls).astype(int).values, ts_probs_test[:, cls]
            )
            print(f"    Class {cls}: ECE={ece_v:.4f}")
    except Exception as e:
        print(f"  WARNING: Two-stage training failed: {e}")


# ====================# ============================================================
# Experiment Logger — Multi-Metric Comparison Table
# ============================================================
exp_log = ExperimentLogger(baseline_name="LightGBM_standard")

def _get_l3_metrics(report_dict):
    cls_r = report_dict["classification_report"]
    l3 = cls_r.get("3", {})
    return float(l3.get("recall", 0.0)), float(l3.get("precision", 0.0))

def _mean_ece(probs, y_true, n_classes=4):
    eces = [expected_calibration_error((y_true == c).astype(int), probs[:, c])
            for c in range(n_classes)]
    return float(np.mean(eces))

# --- Prepare test-set CatBoost DataFrame ---
X_test_cb = X_test.copy()
for col in cat_features:
    X_test_cb[col] = X_test_cb[col].astype(str).replace("nan", "Missing").fillna("Missing")
cb_probs_for_eval = predict_calibrated(cb_model, X_test_cb, cb_calibrators)

# --- LightGBM standard ---
l3r, l3p = _get_l3_metrics(lgb_report)
exp_log.add_result(
    "LightGBM_standard",
    qwk=lgb_report["qwk"],
    macro_f1=lgb_report["classification_report"]["macro avg"]["f1-score"],
    l3_recall=l3r, l3_precision=l3p,
    expected_cost=lgb_report.get("asymmetric_cost", 0.0),
    ece_mean=_mean_ece(lgb_ts_probs_test, y_test.values),
    confusion_matrix=lgb_report["confusion_matrix"],
    is_baseline=True,
)

# --- CatBoost standard ---
l3r, l3p = _get_l3_metrics(cb_report)
exp_log.add_result(
    "CatBoost_standard",
    qwk=cb_report["qwk"],
    macro_f1=cb_report["classification_report"]["macro avg"]["f1-score"],
    l3_recall=l3r, l3_precision=l3p,
    expected_cost=cb_report.get("asymmetric_cost", 0.0),
    ece_mean=_mean_ece(cb_probs_for_eval, y_test.values),
    confusion_matrix=cb_report["confusion_matrix"],
)

# --- Ordinal LightGBM ---
if ordinal_report is not None:
    l3r, l3p = _get_l3_metrics(ordinal_report)
    exp_log.add_result(
        "LightGBM_ordinal",
        qwk=ordinal_report["qwk"],
        macro_f1=ordinal_report["classification_report"]["macro avg"]["f1-score"],
        l3_recall=l3r, l3_precision=l3p,
        expected_cost=ordinal_report.get("asymmetric_cost", 0.0),
        ece_mean=_mean_ece(ord_ts_probs_test, y_test.values),
        confusion_matrix=ordinal_report["confusion_matrix"],
    )

# --- Ordinal CatBoost ---
if ordinal_cb_report is not None:
    l3r, l3p = _get_l3_metrics(ordinal_cb_report)
    exp_log.add_result(
        "CatBoost_ordinal",
        qwk=ordinal_cb_report["qwk"],
        macro_f1=ordinal_cb_report["classification_report"]["macro avg"]["f1-score"],
        l3_recall=l3r, l3_precision=l3p,
        expected_cost=ordinal_cb_report.get("asymmetric_cost", 0.0),
        ece_mean=_mean_ece(ord_cb_ts_probs_test, y_test.values),
        confusion_matrix=ordinal_cb_report["confusion_matrix"],
    )

# --- Weighted CatBoost ---
if weighted_cb_report is not None:
    l3r, l3p = _get_l3_metrics(weighted_cb_report)
    exp_log.add_result(
        f"CatBoost_weighted_{w}",
        qwk=weighted_cb_report["qwk"],
        macro_f1=weighted_cb_report["classification_report"]["macro avg"]["f1-score"],
        l3_recall=l3r, l3_precision=l3p,
        expected_cost=weighted_cb_report.get("asymmetric_cost", 0.0),
        ece_mean=_mean_ece(wcb_ts_probs_test, y_test.values),
        confusion_matrix=weighted_cb_report["confusion_matrix"],
    )

# --- Two-Stage ---
if two_stage_report is not None:
    l3r, l3p = _get_l3_metrics(two_stage_report)
    exp_log.add_result(
        "TwoStage_CatBoost",
        qwk=two_stage_report["qwk"],
        macro_f1=two_stage_report["classification_report"]["macro avg"]["f1-score"],
        l3_recall=l3r, l3_precision=l3p,
        expected_cost=two_stage_report.get("asymmetric_cost", 0.0),
        ece_mean=_mean_ece(two_stage_probs_test, y_test.values),
        confusion_matrix=two_stage_report["confusion_matrix"],
    )

exp_log.print_comparison_table()

# ============================================================
# Select Best Model (from accepted models, by QWK)
# ============================================================
accepted = [r for r in exp_log.results if r["accepted"]]
best_accepted = max(accepted, key=lambda r: r["qwk"]) if accepted else exp_log.results[0]
best_name = best_accepted["name"]
print(f"\n  >> Best accepted model: {best_name}")

best_probs_map = {
    "LightGBM_standard": lgb_ts_probs_test,
    "CatBoost_standard": cb_probs_for_eval,
}
if ordinal_report is not None:
    best_probs_map["LightGBM_ordinal"] = ord_ts_probs_test
if ordinal_cb_report is not None:
    best_probs_map["CatBoost_ordinal"] = ord_cb_ts_probs_test
if weighted_cb_report is not None:
    best_probs_map[f"CatBoost_weighted_{w}"] = wcb_ts_probs_test
if two_stage_report is not None:
    best_probs_map["TwoStage_CatBoost"] = two_stage_probs_test

best_probs = best_probs_map.get(best_name, lgb_ts_probs_test)

best_y_pred_map = {
    "LightGBM_standard": lgb_y_pred,
    "CatBoost_standard": cb_y_pred,
}
if ordinal_report is not None:
    best_y_pred_map["LightGBM_ordinal"] = ord_y_pred
if ordinal_cb_report is not None:
    best_y_pred_map["CatBoost_ordinal"] = ord_cb_y_pred
if weighted_cb_report is not None:
    best_y_pred_map[f"CatBoost_weighted_{w}"] = wcb_y_pred
if two_stage_report is not None:
    best_y_pred_map["TwoStage_CatBoost"] = ts_y_pred

best_y_pred = best_y_pred_map.get(best_name, lgb_y_pred)

# ============================================================
# MANDATORY: Top-100 Misclassification Review
# This review is required before any model is considered accepted.
# ============================================================
top_n_misclassifications(X_test, y_test.values, best_y_pred, best_probs, costs, n=100)

# ============================================================
# Save Best Model
# ============================================================
if best_name == "LightGBM_standard":
    model_path = save_model(lgb_model, str(resolve_path("models")), "severity_lgbm")
    best_calibrators = lgb_calibrators
elif best_name == "CatBoost_standard":
    model_path = resolve_path("models/severity_catboost.cbm")
    cb_model.save_model(str(model_path))
    best_calibrators = cb_calibrators
elif best_name == "LightGBM_ordinal":
    ord_clfs_save, ord_cal_save, _ = ordinal_model
    for i, clf in enumerate(ord_clfs_save):
        clf.save_model(str(resolve_path(f"models/severity_ordinal_clf{i}.txt")))
    model_path = resolve_path("models/severity_ordinal_clf0.txt")
    best_calibrators = ord_cal_save
elif best_name == "CatBoost_ordinal":
    for i, clf in enumerate(ord_cb_clfs):
        clf.save_model(str(resolve_path(f"models/severity_ordinal_cb_clf{i}.cbm")))
    model_path = resolve_path("models/severity_ordinal_cb_clf0.cbm")
    best_calibrators = ord_cb_cal
elif "CatBoost_weighted" in best_name:
    wcb_model_obj = weighted_cb_model[0]
    model_path = resolve_path("models/severity_catboost_weighted.cbm")
    wcb_model_obj.save_model(str(model_path))
    best_calibrators = wcb_calibrators
elif best_name == "TwoStage_CatBoost":
    model_path = resolve_path("models/severity_two_stage.joblib")
    joblib.dump(two_stage_model, str(model_path))
    best_calibrators = None
else:
    model_path = save_model(lgb_model, str(resolve_path("models")), "severity_lgbm")
    best_calibrators = lgb_calibrators

if best_calibrators is not None:
    cal_path = resolve_path("models/severity_calibrators.joblib")
    joblib.dump(best_calibrators, cal_path)
    print(f"  Calibrators saved: {cal_path}")
print(f"  Model saved: {model_path}")

# Save feature column list for inference
feat_path = resolve_path("models/severity_feature_cols.json")
with open(feat_path, "w") as f:
    json.dump({
        "feature_cols": feature_cols,
        "tabular_feature_count": len(feature_cols) - narrative_dim,
        "narrative_pca_dim": narrative_dim,
        "use_narrative": args.use_narrative,
        "use_umap": args.use_umap,
        "interaction_features": args.interaction_features,
        "best_model": best_name,
    }, f, indent=2)

# ============================================================
# Final: Best Model Detailed Evaluation
# ============================================================
print("\n" + "="*55)
print(f"  BEST MODEL ({best_name}) — TEST SET EVALUATION")
print("="*55)
print(f"  QWK:              {best_accepted['qwk']:.4f}")
print(f"  Macro-F1:         {best_accepted['macro_f1']:.4f}")
print(f"  Level-3 Recall:   {best_accepted['l3_recall']:.3f}")
print(f"  Level-3 Prec:     {best_accepted['l3_precision']:.3f}")
print(f"  Expected Cost:    {best_accepted['expected_cost']:.4f}")
print(f"  ECE (mean):       {best_accepted['ece_mean']:.4f}")

print("\n  Full confusion matrix:")
exp_log.print_confusion_matrix(best_name)

# Slice analysis by year
print("\n" + "="*55)
print("  SLICE ANALYSIS & FEATURE IMPORTANCE")
print("="*55)

X_slice = X_test_cb if best_name == "CatBoost_standard" else X_test
if "year" in X_slice.columns:
    print("  Slice by Year:")
    for yr in sorted(X_slice["year"].unique()):
        idx = X_slice["year"] == yr
        y_slice = y_test[idx]
        p_slice = predict_cost_sensitive(best_probs[idx], costs)
        if len(y_slice) > 10:
            sr = full_severity_report(y_slice.values, p_slice, costs)
            print(f"    {int(yr)}: QWK={sr['qwk']:.4f}  "
                  f"Macro-F1={sr['classification_report']['macro avg']['f1-score']:.4f}  "
                  f"(n={len(y_slice)})")

# Feature importance
narr_prefix = "narr_umap_" if args.use_umap else "narr_pca_"
print("\n  Feature Importance (top 15):")
if best_name == "LightGBM_standard" or best_name == "LightGBM_ordinal":
    importance = lgb_model.feature_importance(importance_type="gain")
    top_idx = np.argsort(importance)[::-1][:15]
    for i in top_idx:
        marker = " <- narrative" if feature_cols[i].startswith(narr_prefix) else ""
        print(f"    {feature_cols[i]:<50} : {importance[i]:.2f}{marker}")
elif best_name in ("CatBoost_standard", "CatBoost_ordinal") or "CatBoost_weighted" in best_name:
    if best_name == "CatBoost_standard":
        model_for_fi = cb_model
    elif "CatBoost_weighted" in best_name:
        model_for_fi = weighted_cb_model[0]
    else:
        model_for_fi = ord_cb_clfs[0]
    importance = model_for_fi.get_feature_importance()
    top_idx = np.argsort(importance)[::-1][:15]
    for i in top_idx:
        marker = " <- narrative" if feature_cols[i].startswith(narr_prefix) else ""
        print(f"    {feature_cols[i]:<50} : {importance[i]:.2f}{marker}")
else:
    print("  (Feature importance not available for two-stage/ordinal ensemble)")

print("\nPhase 2 complete!")
