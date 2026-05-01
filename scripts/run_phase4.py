"""
Phase 4 — Problem D: Emerging Risk Discovery
=============================================
Runs unsupervised pattern discovery on ASRS narratives:
  1. BERTopic clustering
  2. Temporal shift detection (PELT)
  3. Emerging risk scoring

Usage:
    python scripts/run_phase4.py
"""

import json
import time
import numpy as np
import pandas as pd

from src.utils.config import load_main_config, resolve_path, set_seeds
from src.data.loader import load_raw_data, parse_time_date
from src.data.target_engineering import apply_severity_rubric
from src.features.temporal import extract_temporal_features
from src.features.text import preprocess_narratives, combine_text_fields
from src.models.discovery import (
    fit_bertopic, get_topic_info, calculate_temporal_trends,
    calculate_temporal_trends_long,
    detect_changepoints, score_emerging_risks, save_discovery_model
)

def banner(text):
    print(f"\n{'='*60}\n  {text}\n{'='*60}")

def main():
    set_seeds()
    banner("Phase 4 — Problem D: Emerging Risk Discovery")

    print("Loading data...")
    config = load_main_config()
    df = load_raw_data(config)
    df = apply_severity_rubric(df)
    df = parse_time_date(df)
    df = extract_temporal_features(df)
    df = preprocess_narratives(df)
    df = combine_text_fields(df, output_col="combined_text")

    # Load cached embeddings from Phase 3 if they exist
    emb_path = resolve_path("data/processed")
    if (emb_path / "emb_train.npy").exists():
        print("Loading cached SBERT embeddings from Phase 3...")
        # To align correctly with df, we should recreate splits or just use the same split logic
        from src.features.temporal import create_temporal_split
        df = create_temporal_split(df)
        
        tr_mask = df['split'] == 'train'
        va_mask = df['split'] == 'val'
        te_mask = df['split'] == 'test'
        
        emb_train = np.load(emb_path / "emb_train.npy")
        emb_val   = np.load(emb_path / "emb_val.npy")
        emb_test  = np.load(emb_path / "emb_test.npy")
        
        # We need a unified array and text series aligned together.
        # Validate that split masks produce the expected counts before stacking.
        n_train, n_val, n_test = tr_mask.sum(), va_mask.sum(), te_mask.sum()
        assert emb_train.shape[0] == n_train, (
            f"Embedding/split mismatch: emb_train has {emb_train.shape[0]} rows, "
            f"but train split has {n_train} rows"
        )
        assert emb_val.shape[0] == n_val, (
            f"Embedding/split mismatch: emb_val has {emb_val.shape[0]} rows, "
            f"but val split has {n_val} rows"
        )
        assert emb_test.shape[0] == n_test, (
            f"Embedding/split mismatch: emb_test has {emb_test.shape[0]} rows, "
            f"but test split has {n_test} rows"
        )
        
        df_ordered = pd.concat([df[tr_mask], df[va_mask], df[te_mask]])
        embeddings = np.vstack([emb_train, emb_val, emb_test])
        
        assert len(df_ordered) == embeddings.shape[0], (
            f"Row count mismatch after concat: df={len(df_ordered)}, "
            f"embeddings={embeddings.shape[0]}"
        )
        texts = df_ordered["combined_text"]
    else:
        print("Cached embeddings not found. Please run Phase 3 Tier 2 first.")
        return

    banner("1. BERTopic Clustering")
    start = time.time()
    bertopic_params = config.get("model_params", {}).get("bertopic", {})
    topic_model, topics = fit_bertopic(
        texts, embeddings,
        min_topic_size=bertopic_params.get("min_topic_size", 50),
    )
    topic_info = get_topic_info(topic_model)
    print(f"  Found {len(topic_info) - 1} active topics (excluding outlier class -1).")
    
    # Save the model
    save_discovery_model(topic_model, resolve_path("models/bertopic_model"))

    banner("2. Temporal Dynamics & Changepoints")
    monthly_counts = calculate_temporal_trends(df_ordered, topics, time_col="Time_Date")
    print(f"  Computed monthly counts. Shape: {monthly_counts.shape}")

    pelt_params = config.get("model_params", {}).get("pelt", {})
    changepoints = detect_changepoints(
        monthly_counts,
        penalty=pelt_params.get("penalty", 3.0),
    )
    num_cps = sum(len(cps) > 0 for cps in changepoints.values())
    print(f"  Detected changepoints in {num_cps} topics.")

    banner("3. Emerging Risk Scoring")
    scores_df = score_emerging_risks(
        topic_info, monthly_counts, changepoints, df_ordered, topics
    )

    # Save results
    scores_path = resolve_path("data/processed/emerging_risks.csv")
    scores_df.to_csv(scores_path, index=False)
    print(f"  Saved risk scores to {scores_path}")

    print("\n  Top 10 Emerging Risks:")
    top_10 = scores_df.head(10)
    for _, row in top_10.iterrows():
        print(f"    Topic {row['Topic']:<3} | Score: {row['Risk_Score']:<6.2f} | Growth: {row['Growth_Ratio']:<5.2f} | "
              f"Recent CP: {str(row['Recent_Changepoint']):<5} | Name: {row['Name']}")

    # ──────────────────────────────────────────────────────────
    # 4. Save Streamlit-required artefacts
    # ──────────────────────────────────────────────────────────
    banner("4. Saving Streamlit Artefacts")

    # topic_trends.parquet — long format with per-period avg_severity
    trends_long = calculate_temporal_trends_long(df_ordered, topics, time_col="Time_Date")
    trends_path = resolve_path("data/processed/topic_trends.parquet")
    trends_long.to_parquet(trends_path, index=False)
    print(f"  Saved topic_trends.parquet  ({len(trends_long)} rows)")

    # topic_representations.json — {topic_id: {keywords: [...]}}
    reps: dict = {}
    for _, trow in topic_info.iterrows():
        tid = int(trow["Topic"])
        if tid == -1:
            continue
        # BERTopic stores top-N (word, weight) tuples per topic
        raw_kws = topic_model.get_topic(tid)
        keywords = [w for w, _ in raw_kws] if raw_kws else []
        reps[str(tid)] = {"keywords": keywords}
    reps_path = resolve_path("data/processed/topic_representations.json")
    with open(reps_path, "w") as f:
        json.dump(reps, f, indent=2)
    print(f"  Saved topic_representations.json  ({len(reps)} topics)")

    # topic_changepoints.json — {topic_id: [period_indices]}
    cps_out = {
        str(int(k)): [int(x) for x in v]
        for k, v in changepoints.items()
    }
    cp_path = resolve_path("data/processed/topic_changepoints.json")
    with open(cp_path, "w") as f:
        json.dump(cps_out, f, indent=2)
    print(f"  Saved topic_changepoints.json  ({len(cps_out)} topics)")

    elapsed = time.time() - start
    print(f"\n  Phase 4 done in {elapsed:.1f}s")

if __name__ == "__main__":
    main()
