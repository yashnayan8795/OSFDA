"""
Phase 3 — Problem B: Category Classification
=============================================
Runs three tiers:
  1. TF-IDF + Logistic Regression (baseline)
  2. Sentence-BERT text tower (ablation)
  3. Fusion: Text Tower + LightGBM tabular embeddings → MLP

Outputs a comparison table and saves all models.
Usage:
    python scripts/run_phase3.py [--tier {1,2,3,all}]
"""

import argparse
import time
import numpy as np
import pandas as pd
import joblib

from src.utils.config import (
    load_main_config, load_feature_whitelist,
    resolve_path, set_seeds,
)
from src.data.loader import load_raw_data, parse_time_date
from src.data.target_engineering import (
    apply_severity_rubric, apply_category_taxonomy,
)
from src.data.leakage_audit import get_problem_b_features, get_problem_a_features
from src.features.temporal import (
    extract_temporal_features, create_temporal_split, get_split_data,
)
from src.features.encoding import (
    identify_column_types, prepare_for_lgbm, bucket_experience,
)
from src.features.text import preprocess_narratives, combine_text_fields
from src.models.category import (
    build_tfidf_baseline, predict_tfidf,
    encode_texts, build_text_tower, predict_text_tower,
    FusionMLP, get_lgbm_leaf_embeddings,
    save_category_model,
)
from src.evaluation.multilabel_metrics import multilabel_report


def banner(text):
    print(f"\n{'='*60}\n  {text}\n{'='*60}")


def load_and_prepare():
    """Shared data loading for all tiers."""
    config = load_main_config()
    df = load_raw_data(config)
    df = apply_severity_rubric(df)
    df = parse_time_date(df)
    df = create_temporal_split(df)
    df = extract_temporal_features(df)
    df = bucket_experience(df)

    # Rebuild category targets with current taxonomy
    df, cat_matrix = apply_category_taxonomy(df)
    label_names = sorted(cat_matrix.columns.tolist())

    # Preprocess text
    df = preprocess_narratives(df)
    df = combine_text_fields(df, output_col="combined_text")

    return df, cat_matrix, label_names


def get_splits(df, cat_matrix, label_names):
    """Return (X_text, y_labels, tabular_X) dicts for train/val/test."""
    splits_info = {}
    for sp in ["train", "val", "test"]:
        mask = df["split"] == sp
        splits_info[sp] = {
            "text": df.loc[mask, "combined_text"],
            "y": cat_matrix.loc[mask],
            "df": df[mask],
        }
    return splits_info


def print_report(name, report, label_names):
    print(f"\n  {name}:")
    print(f"    Macro-F1:    {report['macro_f1']:.4f}")
    print(f"    Micro-F1:    {report['micro_f1']:.4f}")
    print(f"    Hamming:     {report['hamming_loss']:.4f}")
    print(f"    Subset Acc:  {report['subset_accuracy']:.4f}")
    print("    Per-label F1:")
    for label in label_names:
        f1 = report["per_label"][label]["f1"]
        sup = report["per_label"][label]["support"]
        print(f"      {label:<25} F1={f1:.3f}  (n={sup})")


def run_tier1(splits, label_names):
    banner("TIER 1 — TF-IDF Baseline")
    start = time.time()

    model = build_tfidf_baseline(
        splits["train"]["text"], splits["train"]["y"],
        splits["val"]["text"],   splits["val"]["y"],
        label_names,
    )
    print(f"  Per-label thresholds: {model['thresholds']}")

    # Val evaluation
    _, val_preds = predict_tfidf(model, splits["val"]["text"])
    val_report = multilabel_report(
        splits["val"]["y"].values, val_preds, label_names
    )
    print_report("Validation", val_report, label_names)

    # Test evaluation
    _, test_preds = predict_tfidf(model, splits["test"]["text"])
    test_report = multilabel_report(
        splits["test"]["y"].values, test_preds, label_names
    )
    print_report("Test", test_report, label_names)

    save_category_model(model, resolve_path("models/category_tfidf_baseline.joblib"))
    elapsed = time.time() - start
    print(f"\n  Done in {elapsed:.1f}s")
    return model, test_report


def run_tier2(splits, label_names):
    banner("TIER 2 — Sentence-BERT Text Tower (ablation)")
    start = time.time()

    # Encode texts
    print("  Encoding train texts...")
    emb_train = encode_texts(splits["train"]["text"], batch_size=128)
    print("  Encoding val texts...")
    emb_val   = encode_texts(splits["val"]["text"],   batch_size=128)
    print("  Encoding test texts...")
    emb_test  = encode_texts(splits["test"]["text"],  batch_size=128)

    print(f"  Embedding dim: {emb_train.shape[1]}")

    # Save embeddings for reuse
    emb_path = resolve_path("data/processed")
    np.save(emb_path / "emb_train.npy", emb_train)
    np.save(emb_path / "emb_val.npy",   emb_val)
    np.save(emb_path / "emb_test.npy",  emb_test)
    print("  Embeddings saved.")

    model = build_text_tower(
        emb_train, splits["train"]["y"],
        emb_val,   splits["val"]["y"],
        label_names,
    )
    print(f"  Per-label thresholds: {model['thresholds']}")

    # Val
    _, val_preds = predict_text_tower(model, emb_val)
    val_report = multilabel_report(splits["val"]["y"].values, val_preds, label_names)
    print_report("Validation", val_report, label_names)

    # Test
    _, test_preds = predict_text_tower(model, emb_test)
    test_report = multilabel_report(splits["test"]["y"].values, test_preds, label_names)
    print_report("Test", test_report, label_names)

    save_category_model(model, resolve_path("models/category_text_tower.joblib"))
    elapsed = time.time() - start
    print(f"\n  Done in {elapsed:.1f}s")
    return model, test_report, (emb_train, emb_val, emb_test)


def run_tier3(splits, label_names, embeddings):
    banner("TIER 3 — Fusion Model (Text + Tabular)")
    start = time.time()
    emb_train, emb_val, emb_test = embeddings

    # Build tabular features (Problem B whitelist)
    whitelist = load_feature_whitelist()
    prob_b_feats = get_problem_b_features(splits["train"]["df"], whitelist)
    # Use narrative+event columns as tabular signal too
    text_feature_cols = ["Events_Anomaly", "Events.3_Detector", "Events.4_When Detected"]
    tabular_cols = [c for c in prob_b_feats if c in splits["train"]["df"].columns
                    and c not in text_feature_cols and c != "Time_Date"]
    tabular_cols = [c for c in tabular_cols if splits["train"]["df"][c].nunique() > 1]

    print(f"  Tabular features: {len(tabular_cols)}")

    # Prepare LightGBM tabular encoding (reuse severity model's leaf embeddings)
    import lightgbm as lgb
    from src.features.encoding import identify_column_types, prepare_for_lgbm
    from src.models.severity import get_default_lgbm_params, build_lgbm_dataset

    col_types = identify_column_types(splits["train"]["df"][tabular_cols])
    cat_features = [c for c in col_types["categorical"] if c in tabular_cols]

    # Encode categories for all splits
    all_splits_dfs = {}
    for sp in ["train", "val", "test"]:
        sp_df = prepare_for_lgbm(
            splits[sp]["df"][tabular_cols],
            col_types["categorical"], col_types["numeric"], col_types["medium_missing"]
        )
        all_splits_dfs[sp] = sp_df

    # Train a LightGBM tabular encoder per label (binary, for leaf features)
    # Simpler: use frequency encoding of categoricals → direct feature matrix
    from sklearn.preprocessing import OrdinalEncoder
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_cols_present = [c for c in cat_features if c in tabular_cols]
    num_cols_present  = [c for c in tabular_cols if c not in cat_cols_present]

    def encode_split(sp_df):
        feats = []
        if cat_cols_present:
            cat_enc = encoder.transform(sp_df[cat_cols_present].astype(str).fillna("__missing__"))
            feats.append(cat_enc)
        if num_cols_present:
            num_arr = sp_df[num_cols_present].fillna(0).values.astype(np.float32)
            feats.append(num_arr)
        return np.hstack(feats) if feats else np.zeros((len(sp_df), 1))

    # Fit encoder on train
    if cat_cols_present:
        encoder.fit(all_splits_dfs["train"][cat_cols_present].astype(str).fillna("__missing__"))

    tab_train = encode_split(all_splits_dfs["train"])
    tab_val   = encode_split(all_splits_dfs["val"])
    tab_test  = encode_split(all_splits_dfs["test"])
    print(f"  Tabular encoding shape: {tab_train.shape}")

    # Fusion model
    print("  Training Fusion MLP heads...")
    fusion = FusionMLP(hidden_dims=(256, 128), max_epochs=20)
    fusion.fit(
        emb_train, tab_train, splits["train"]["y"],
        emb_val,   tab_val,   splits["val"]["y"],
        label_names,
    )
    print(f"  Per-label thresholds: {fusion.thresholds}")

    # Val
    _, val_preds = fusion.predict(emb_val, tab_val)
    val_report = multilabel_report(splits["val"]["y"].values, val_preds, label_names)
    print_report("Validation", val_report, label_names)

    # Test
    _, test_preds = fusion.predict(emb_test, tab_test)
    test_report = multilabel_report(splits["test"]["y"].values, test_preds, label_names)
    print_report("Test", test_report, label_names)

    # Save fusion model AND the encoder + column metadata for inference
    fusion_artifact = {
        "fusion_model": fusion,
        "ordinal_encoder": encoder if cat_cols_present else None,
        "cat_cols": cat_cols_present,
        "num_cols": num_cols_present,
        "tabular_cols": tabular_cols,
    }
    save_category_model(fusion, resolve_path("models/category_fusion.joblib"))
    save_category_model(fusion_artifact, resolve_path("models/category_fusion_full.joblib"))
    elapsed = time.time() - start
    print(f"\n  Done in {elapsed:.1f}s")
    return fusion, test_report


def print_comparison(results):
    banner("PHASE 3 — MODEL COMPARISON (Test Set)")
    print(f"  {'Model':<30} {'Macro-F1':>10} {'Micro-F1':>10} {'Hamming':>10}")
    print("  " + "-"*65)
    for name, r in results.items():
        print(
            f"  {name:<30} {r['macro_f1']:>10.4f} {r['micro_f1']:>10.4f} {r['hamming_loss']:>10.4f}"
        )
    # Gate check
    best_macro = max(r['macro_f1'] for r in results.values())
    gate = "PASSED" if best_macro >= 0.55 else "BELOW TARGET (0.55)"
    print(f"\n  G4 Quality Gate (Macro-F1 >= 0.55): {gate}")
    print(f"  Best Macro-F1: {best_macro:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3 category model")
    parser.add_argument("--tier", choices=["1", "2", "3", "all"], default="all")
    args = parser.parse_args()

    set_seeds()
    banner("Phase 3 — Problem B: Incident Category Classification")

    print("Loading and preparing data...")
    df, cat_matrix, label_names = load_and_prepare()
    splits = get_splits(df, cat_matrix, label_names)

    print(f"  Labels: {label_names}")
    print(f"  Train: {len(splits['train']['text'])}, Val: {len(splits['val']['text'])}, Test: {len(splits['test']['text'])}")

    results = {}
    embeddings = None

    run_all = args.tier == "all"

    if run_all or args.tier == "1":
        _, r = run_tier1(splits, label_names)
        results["TF-IDF Baseline"] = r

    if run_all or args.tier in ("2", "3"):
        # Check for cached embeddings first
        emb_path = resolve_path("data/processed")
        if (emb_path / "emb_train.npy").exists() and args.tier in ("3", "all"):
            print("\n  Loading cached embeddings...")
            emb_train = np.load(emb_path / "emb_train.npy")
            emb_val   = np.load(emb_path / "emb_val.npy")
            emb_test  = np.load(emb_path / "emb_test.npy")
            embeddings = (emb_train, emb_val, emb_test)
            print(f"  Loaded cached embeddings, shape: {emb_train.shape}")
        else:
            _, r, embeddings = run_tier2(splits, label_names)
            results["Text Tower (SBERT)"] = r

    if run_all or args.tier == "3":
        if embeddings is None:
            emb_path = resolve_path("data/processed")
            emb_train = np.load(emb_path / "emb_train.npy")
            emb_val   = np.load(emb_path / "emb_val.npy")
            emb_test  = np.load(emb_path / "emb_test.npy")
            embeddings = (emb_train, emb_val, emb_test)
        _, r = run_tier3(splits, label_names, embeddings)
        results["Fusion (Text + Tabular)"] = r

    if results:
        print_comparison(results)


if __name__ == "__main__":
    main()
