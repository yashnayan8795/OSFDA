"""
Flight Risk Analysis — End-to-End Pipeline Runner
===================================================
Orchestrates: Data Acquisition → Target Engineering → Leakage Audit →
Temporal Split → Feature Engineering → Model Training → Evaluation.

Run from project root:
    python -m src.pipeline
"""

import sys
import time
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.config import (
    load_main_config, load_severity_rubric, load_category_taxonomy,
    load_feature_whitelist, load_cost_matrix, resolve_path, set_seeds, PROJECT_ROOT,
)
from src.data.loader import (
    download_asrs_dataset, save_raw_data, load_raw_data,
    column_inventory, parse_time_date,
)
from src.data.target_engineering import (
    apply_severity_rubric, validate_severity_distribution,
    apply_category_taxonomy, rubric_hash,
)
from src.data.leakage_audit import (
    get_problem_a_features, validate_no_leakage, filter_features,
)
from src.features.temporal import (
    extract_temporal_features, create_temporal_split,
    validate_temporal_split, get_split_data,
)
from src.features.encoding import (
    identify_column_types, prepare_for_lgbm, bucket_experience,
)
from src.features.text import preprocess_narratives, redaction_stats


def banner(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def run_phase0():
    """Phase 0: Verify scaffolding and configuration."""
    banner("PHASE 0 — Scaffolding Verification")
    config = load_main_config()
    print(f"  Project: {config['project']['name']} v{config['project']['version']}")
    print(f"  Root: {PROJECT_ROOT}")

    rubric = load_severity_rubric()
    print(f"  Severity rubric: v{rubric['version']} ({len(rubric['levels'])} levels)")

    taxonomy = load_category_taxonomy()
    print(f"  Category taxonomy: v{taxonomy['version']} ({len(taxonomy['categories'])} categories)")

    whitelist = load_feature_whitelist()
    print(f"  Feature whitelist: v{whitelist['version']}")
    print(f"    Problem A: {len(whitelist['problem_a_whitelist'])} features")
    print(f"    Problem B extras: {len(whitelist['problem_b_extra'])} features")

    cost = load_cost_matrix()
    print(f"  Cost matrix: v{cost['version']}")

    print("\n  ✅ Phase 0 passed: All configs load successfully.")
    return config


def run_phase1(config):
    """Phase 1: Data Foundation — acquisition, targets, leakage audit, split."""
    banner("PHASE 1 — Data Foundation")

    # 1.1 Data Acquisition
    print("\n  [1.1] Data Acquisition")
    df = load_raw_data(config)
    print(f"    Shape: {df.shape}")

    inv = column_inventory(df)
    inv_path = resolve_path("data/raw/column_inventory.csv")
    inv.to_csv(inv_path, index=False)
    print(f"    Column inventory saved to {inv_path}")

    # 1.2 Target Engineering — Severity
    print("\n  [1.2] Severity Rubric (v1)")
    df = apply_severity_rubric(df)
    validation = validate_severity_distribution(df)
    print(f"    Rubric hash: {rubric_hash()}")
    print(f"    Distribution: {validation['distribution']}")
    if validation["is_valid"]:
        print("    ✅ Distribution within expected ranges.")
    else:
        for w in validation["warnings"]:
            print(f"    ⚠️  {w}")

    # Save severity targets
    sev_path = resolve_path("data/processed/severity_targets.parquet")
    sev_path.parent.mkdir(parents=True, exist_ok=True)
    df[["acn_num_ACN", "severity_level"]].to_parquet(sev_path, index=False)
    print(f"    Targets saved: {sev_path}")

    # 1.3 Target Engineering — Category
    print("\n  [1.3] Category Taxonomy (v1)")
    df, cat_matrix = apply_category_taxonomy(df)
    print(f"    Category counts:\n{cat_matrix.sum().to_string()}")
    avg_labels = cat_matrix.sum(axis=1).mean()
    zero_labels = (cat_matrix.sum(axis=1) == 0).sum()
    print(f"    Avg labels/report: {avg_labels:.2f}")
    print(f"    Reports with zero labels: {zero_labels} ({zero_labels/len(df)*100:.1f}%)")

    cat_path = resolve_path("data/processed/category_targets.parquet")
    cat_out = pd.concat([df[["acn_num_ACN", "primary_category"]], cat_matrix], axis=1)
    cat_out.to_parquet(cat_path, index=False)
    print(f"    Targets saved: {cat_path}")

    # 1.4 Leakage Audit
    print("\n  [1.4] Leakage Audit")
    whitelist = load_feature_whitelist()
    prob_a_feats = get_problem_a_features(df, whitelist)
    is_clean, leaks = validate_no_leakage(prob_a_feats, problem="A", whitelist=whitelist)
    print(f"    Problem A features: {len(prob_a_feats)}")
    if is_clean:
        print("    ✅ No leakage detected in Problem A features.")
    else:
        for leak in leaks:
            print(f"    ❌ LEAK: {leak}")

    # 1.5 Temporal Split
    print("\n  [1.5] Temporal Split")
    df = parse_time_date(df)
    df = create_temporal_split(df)
    split_info = validate_temporal_split(df)
    for s in ["train", "val", "test"]:
        info = split_info[s]
        print(f"    {s}: {info['count']} ({info['pct']}%) — years {info['year_min']}-{info['year_max']}")
    if split_info["is_valid"]:
        print("    ✅ No temporal overlap.")
    else:
        print("    ❌ OVERLAP DETECTED!")

    split_path = resolve_path("data/processed/temporal_splits.parquet")
    df[["acn_num_ACN", "year", "month", "split"]].to_parquet(split_path, index=False)
    print(f"    Splits saved: {split_path}")

    return df, cat_matrix


def run_phase2(df, config):
    """Phase 2: Problem A — Severity Model Training & Evaluation."""
    banner("PHASE 2 — Problem A: Severity Classification")

    from src.models.severity import train_lgbm_severity, calibrate_model, predict_calibrated, save_model
    from src.evaluation.ordinal_metrics import full_severity_report
    from src.evaluation.calibration import expected_calibration_error

    whitelist = load_feature_whitelist()

    # Feature engineering
    print("\n  [2.1] Feature Engineering")
    df = extract_temporal_features(df)
    df = bucket_experience(df)

    prob_a_feats = get_problem_a_features(df, whitelist)
    # Add derived features
    derived = ["year", "month", "quarter", "month_sin", "month_cos", "time_of_day_bucket"]
    if "experience_bucket" in df.columns:
        derived.append("experience_bucket")
    feature_cols = list(set(prob_a_feats + [d for d in derived if d in df.columns]))
    # Remove Time_Date (raw) — we use derived features instead
    feature_cols = [c for c in feature_cols if c != "Time_Date"]

    col_types = identify_column_types(df[feature_cols])
    df_prepped = prepare_for_lgbm(df, col_types["categorical"], col_types["numeric"], col_types["medium_missing"])

    # Drop high-missing columns
    for hm in col_types["high_missing"]:
        if hm in feature_cols:
            feature_cols.remove(hm)
            print(f"    Dropped (>80% missing): {hm}")

    print(f"    Final feature count: {len(feature_cols)}")
    print(f"    Categoricals: {len(col_types['categorical'])}")
    print(f"    Numerics: {len(col_types['numeric'])}")

    # Split
    splits = get_split_data(df_prepped, "severity_level", feature_cols)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    cat_features = [c for c in col_types["categorical"] if c in feature_cols]

    # Train
    print("\n  [2.2] Training LightGBM")
    model, history = train_lgbm_severity(X_train, y_train, X_val, y_val, cat_features)
    print(f"    Best iteration: {history['best_iteration']}")

    # Calibrate
    print("\n  [2.3] Isotonic Calibration")
    calibrators = calibrate_model(model, X_val, y_val)

    # Evaluate on test
    print("\n  [2.4] Test Set Evaluation")
    cal_probs = predict_calibrated(model, X_test, calibrators)
    y_pred = cal_probs.argmax(axis=1)

    cost_config = load_cost_matrix()
    report = full_severity_report(y_test, y_pred, cost_config.get("costs"))
    print(f"    QWK: {report['qwk']:.4f}")
    print(f"    QWK 95% CI: [{report['qwk_bootstrap']['ci_low']:.4f}, {report['qwk_bootstrap']['ci_high']:.4f}]")
    print(f"    Ordinal MAE: {report['ordinal_mae']:.4f}")
    if "asymmetric_cost" in report:
        print(f"    Asymmetric Cost: {report['asymmetric_cost']:.4f}")

    # ECE per class
    for cls in range(4):
        ece = expected_calibration_error(
            (y_test == cls).astype(int).values, cal_probs[:, cls]
        )
        print(f"    ECE (class {cls}): {ece:.4f}")

    # Save model
    model_path = save_model(model, str(resolve_path("models")), "severity_lgbm")
    print(f"\n    Model saved: {model_path}")

    return model, calibrators, report


def run_phase3():
    """Phase 3: Problem B — Category Classification."""
    banner("PHASE 3 — Problem B: Category Classification")
    print("  Running scripts/run_phase3.py (TF-IDF Baseline & SBERT Text Tower)...")
    # Using tier 1 and tier 2 since tier 3 was unstable on the first pass
    cmd = [sys.executable, "scripts/run_phase3.py", "--tier", "all"]
    subprocess.run(cmd, check=True)


def run_phase4():
    """Phase 4: Problem D — Emerging Risk Discovery."""
    banner("PHASE 4 — Problem D: Emerging Risk Discovery")
    print("  Running scripts/run_phase4.py (BERTopic & Changepoint Detection)...")
    cmd = [sys.executable, "scripts/run_phase4.py"]
    subprocess.run(cmd, check=True)


def main():
    """Run the full pipeline."""
    start = time.time()
    banner("Flight Risk Analysis — Full Pipeline")
    set_seeds()

    config = run_phase0()
    df, cat_matrix = run_phase1(config)
    model, calibrators, report = run_phase2(df, config)
    run_phase3()
    run_phase4()

    elapsed = time.time() - start
    banner(f"PIPELINE COMPLETE — {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
