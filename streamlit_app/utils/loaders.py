"""
Cached model and data loaders shared across all Streamlit pages.
All heavy IO is wrapped in @st.cache_resource or @st.cache_data.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODELS = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"


# ──────────────────────────────────────────────────────────────
# Problem A — Severity
# ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_severity_model():
    import lightgbm as lgb
    import joblib
    model = lgb.Booster(model_file=str(MODELS / "severity_lgbm.txt"))
    calibrators = joblib.load(MODELS / "severity_calibrators.joblib")
    return model, calibrators


@st.cache_data
def load_severity_test_predictions():
    """Preprocess ASRS data and run the severity model on the test split."""
    from src.data.loader import load_raw_data
    from src.data.target_engineering import apply_severity_rubric
    from src.data.leakage_audit import get_problem_a_features
    from src.features.temporal import extract_temporal_features, create_temporal_split, get_split_data
    from src.features.encoding import identify_column_types, prepare_for_lgbm, bucket_experience
    from src.utils.config import load_main_config, load_feature_whitelist

    config = load_main_config()
    whitelist = load_feature_whitelist()

    df = load_raw_data(config)
    df = apply_severity_rubric(df)
    df = _parse_time_date(df)
    df = create_temporal_split(df)
    df = extract_temporal_features(df)
    df = bucket_experience(df)

    prob_a_feats = get_problem_a_features(df, whitelist)
    derived = ["year", "month", "quarter", "month_sin", "month_cos", "time_of_day_bucket"]
    if "experience_bucket" in df.columns:
        derived.append("experience_bucket")
    feature_cols = list(set(prob_a_feats + [d for d in derived if d in df.columns]))
    feature_cols = [c for c in feature_cols if c != "Time_Date"]

    col_types = identify_column_types(df[feature_cols])
    for hm in col_types["high_missing"]:
        if hm in feature_cols:
            feature_cols.remove(hm)

    df = prepare_for_lgbm(df, col_types["categorical"], col_types["numeric"], col_types["medium_missing"])
    splits = get_split_data(df, "severity_level", feature_cols)
    X_test, y_test = splits["test"]

    model, calibrators = load_severity_model()
    raw_probs = model.predict(X_test)
    cal_probs = np.column_stack([c.predict(raw_probs[:, i]) for i, c in enumerate(calibrators)])
    row_sums = cal_probs.sum(axis=1, keepdims=True)
    cal_probs = cal_probs / np.where(row_sums == 0, 1, row_sums)
    y_pred = cal_probs.argmax(axis=1)

    return y_test.values, y_pred, cal_probs, X_test, feature_cols


def _parse_time_date(df):
    df = df.copy()
    td = pd.to_numeric(df["Time_Date"], errors="coerce").astype("Int64")
    df["year"] = td // 100
    df["month"] = td % 100
    return df


# ──────────────────────────────────────────────────────────────
# Problem B — Category
# ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_tfidf_model():
    import joblib
    return joblib.load(MODELS / "category_tfidf_baseline.joblib")


@st.cache_resource
def load_sbert_classifier():
    import joblib
    return joblib.load(MODELS / "category_text_tower.joblib")


@st.cache_data
def load_category_test_data():
    """Returns (y_true_df, label_names) for the test split."""
    splits = pd.read_parquet(PROCESSED / "temporal_splits.parquet")
    cats = pd.read_parquet(PROCESSED / "category_targets.parquet")
    label_cols = [c for c in cats.columns if c not in ["acn_num_ACN", "primary_category"]]

    merged = splits.merge(cats, on="acn_num_ACN", how="inner")
    test_df = merged[merged["split"] == "test"]
    return test_df[label_cols], label_cols


# ──────────────────────────────────────────────────────────────
# Problem C — Pre-flight Risk
# ──────────────────────────────────────────────────────────────

_PREFLIGHT_DEFAULT_FEATURES = [
    "month_sin", "month_cos", "dow_sin", "dow_cos", "hour",
    "temp", "rhum", "prcp", "wspd", "pres",
    "airport_risk_rate", "carrier_risk_rate", "route_risk_rate",
    "DISTANCE",
]


@st.cache_resource
def load_preflight_model():
    import joblib
    artifact = joblib.load(MODELS / "preflight_lgbm_calibrated.joblib")
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"]
    return artifact


@st.cache_data
def load_preflight_features():
    """Return the feature list saved alongside the preflight model."""
    import joblib
    artifact = joblib.load(MODELS / "preflight_lgbm_calibrated.joblib")
    if isinstance(artifact, dict) and "features" in artifact:
        return artifact["features"]
    return _PREFLIGHT_DEFAULT_FEATURES


@st.cache_data
def load_preflight_test_data():
    import joblib
    df = pd.read_parquet(PROCESSED / "preflight_features_final.parquet")
    artifact = joblib.load(MODELS / "preflight_lgbm_calibrated.joblib")
    if isinstance(artifact, dict) and "features" in artifact:
        features = [f for f in artifact["features"] if f in df.columns]
    else:
        features = [f for f in _PREFLIGHT_DEFAULT_FEATURES if f in df.columns]

    # Resolve year column — BTS data may store date as FL_DATE or year directly
    if "year" not in df.columns:
        for date_col in ("FL_DATE", "fl_date", "date", "Date"):
            if date_col in df.columns:
                df["year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year
                break
        else:
            # Fall back: use the full dataset if year cannot be determined
            df["year"] = 2020

    test_df = df[df["year"] == 2020].copy()

    # Ensure "incident" target column exists
    if "incident" not in test_df.columns:
        raise KeyError(
            "'incident' column not found in preflight_features_final.parquet. "
            "Re-run scripts/run_phase3.py to regenerate the data."
        )

    return test_df[features + ["incident"]], features


# ──────────────────────────────────────────────────────────────
# Problem D — Emerging Risks
# ──────────────────────────────────────────────────────────────

@st.cache_data
def load_emerging_risks():
    risks = pd.read_csv(PROCESSED / "emerging_risks.csv")
    trends = pd.read_parquet(PROCESSED / "topic_trends.parquet")
    with open(PROCESSED / "topic_representations.json") as f:
        reps = json.load(f)
    with open(PROCESSED / "topic_changepoints.json") as f:
        changepoints = json.load(f)
    return risks, trends, reps, changepoints


# ──────────────────────────────────────────────────────────────
# Problem E — Factor Graph
# ──────────────────────────────────────────────────────────────

@st.cache_data
def load_graph_data():
    import networkx as nx
    centrality = pd.read_csv(PROCESSED / "centrality_report.csv")
    with open(PROCESSED / "factor_graph.json") as f:
        graph_data = json.load(f)
    G = nx.node_link_graph(graph_data, edges="links")
    return G, centrality


@st.cache_data
def load_factor_patterns():
    """Load pre-computed factor co-occurrence patterns.
    Returns [] if the file has not been generated yet (run_phase5.py needed).
    """
    patterns_path = PROCESSED / "factor_patterns.json"
    if not patterns_path.exists():
        return []
    with open(patterns_path) as f:
        return json.load(f)
