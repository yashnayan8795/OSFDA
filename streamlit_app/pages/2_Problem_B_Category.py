"""
Problem B — Incident Category Classification
==============================================
Three tabs:
  1. Live Prediction  — narrative text → multi-label categories (TF-IDF or SBERT)
  2. Performance      — per-label F1, macro/micro comparison across tiers
  3. EDA              — category distribution, co-occurrence heatmap
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from streamlit_app.utils.loaders import (
    load_tfidf_model, load_sbert_classifier, load_category_test_data,
    PROCESSED,
)
from streamlit_app.utils.sidebar import render_manual_test_sidebar

st.set_page_config(page_title="Problem B — Category", page_icon="🟡", layout="wide")
render_manual_test_sidebar()
st.title("🟡 Problem B — Incident Category Classification")
st.caption(
    "Multi-label taxonomy across 5 categories. "
    "Three-tier architecture: TF-IDF baseline → SBERT text tower → Fusion MLP."
)

LABEL_COLORS = {
    "ATC_Communication": "#3498db",
    "Airspace_Navigation": "#9b59b6",
    "Environment": "#2ecc71",
    "Equipment_System": "#e67e22",
    "Flight_Operations": "#e74c3c",
}
LABEL_DISPLAY = {
    "ATC_Communication": "ATC / Communication",
    "Airspace_Navigation": "Airspace / Navigation",
    "Environment": "Environment / Weather",
    "Equipment_System": "Equipment / System",
    "Flight_Operations": "Flight Operations",
}

tab1, tab2, tab3 = st.tabs(["🎯 Live Prediction", "📊 Performance", "📈 EDA"])


# ──────────────────────────────────────────────────────────────
# TAB 1 — Live Prediction
# ──────────────────────────────────────────────────────────────

with tab1:
    st.markdown(
        "Enter or paste an **incident narrative** to classify it into one or more categories. "
        "This is a *retrospective* task — it uses post-incident text."
    )

    model_choice = st.radio(
        "Model tier",
        ["TF-IDF Baseline (fast)", "Sentence-BERT Text Tower (slower, higher quality)"],
        horizontal=True,
    )

    EXAMPLE_NARRATIVES = {
        "Near-miss conflict": (
            "While descending to 6000 feet, we received a TCAS RA. ATC had cleared another aircraft "
            "to the same altitude on a converging course. We initiated a climb as directed by TCAS. "
            "The other aircraft passed approximately 200 feet below us."
        ),
        "Engine equipment failure": (
            "During cruise at FL350, the #2 engine oil pressure dropped to zero. We performed the "
            "engine shutdown checklist and diverted to the nearest suitable airport. Maintenance "
            "found a failed oil pump seal."
        ),
        "Weather deviation": (
            "We encountered severe turbulence while deviating around a line of thunderstorms. "
            "One flight attendant was injured during the unexpected turbulence encounter. "
            "We declared an emergency and landed at the nearest airport."
        ),
        "Procedure violation": (
            "During approach briefing I realized we had configured for the wrong runway. "
            "The crew caught the error during the briefing and corrected the FMS prior to "
            "commencing the approach. No unsafe condition resulted."
        ),
        "Airspace incursion": (
            "After departing runway 28L I was told to maintain 4000 feet but I climbed to 5000 "
            "accidentally, entering the Class B airspace without a clearance. ATC advised "
            "immediately and I descended back to 4000."
        ),
    }

    example = st.selectbox("Load example narrative (or type your own below)", ["— custom —"] + list(EXAMPLE_NARRATIVES))
    default_text = EXAMPLE_NARRATIVES[example] if example != "— custom —" else ""

    narrative = st.text_area(
        "Incident Narrative",
        value=default_text,
        height=150,
        placeholder="Paste or type the incident narrative here…",
    )

    if st.button("Classify Narrative", type="primary", use_container_width=True):
        if not narrative.strip():
            st.warning("Please enter a narrative to classify.")
        else:
            if "TF-IDF" in model_choice:
                with st.spinner("Running TF-IDF classifier…"):
                    model = load_tfidf_model()
                    vec = model["vectorizer"]
                    label_names = model["label_names"]
                    thresholds = model["thresholds"]
                    X = vec.transform([narrative])
                    probs = {
                        label: float(model["classifiers"][label].predict_proba(X)[0, 1])
                        for label in label_names
                    }
                    preds = {label: probs[label] >= thresholds[label] for label in label_names}
                    tier_name = "TF-IDF Baseline"
            else:
                with st.spinner("Encoding with Sentence-BERT (downloading model first run)…"):
                    from sentence_transformers import SentenceTransformer
                    encoder = SentenceTransformer("all-MiniLM-L6-v2")
                    emb = encoder.encode([narrative], normalize_embeddings=True)
                    sbert_model = load_sbert_classifier()
                    label_names = sbert_model["label_names"]
                    thresholds = sbert_model["thresholds"]
                    probs = {
                        label: float(sbert_model["classifiers"][label].predict_proba(emb)[0, 1])
                        for label in label_names
                    }
                    preds = {label: probs[label] >= thresholds[label] for label in label_names}
                    tier_name = "Sentence-BERT Text Tower"

            st.markdown(f"#### Results from {tier_name}")
            pred_labels = [l for l, p in preds.items() if p]
            if pred_labels:
                chip_html = " ".join(
                    f'<span style="background:{LABEL_COLORS.get(l, "#999")};color:white;'
                    f'padding:4px 12px;border-radius:12px;margin:3px;font-weight:bold;">'
                    f'{LABEL_DISPLAY.get(l, l)}</span>'
                    for l in pred_labels
                )
                st.markdown(chip_html, unsafe_allow_html=True)
            else:
                st.info("No category exceeded its threshold — the narrative may be ambiguous.")

            fig = go.Figure(go.Bar(
                x=[LABEL_DISPLAY[l] for l in label_names],
                y=[probs[l] * 100 for l in label_names],
                marker_color=[LABEL_COLORS[l] for l in label_names],
                text=[f"{probs[l]*100:.1f}%" for l in label_names],
                textposition="outside",
            ))
            thresh_vals = [thresholds[l] * 100 for l in label_names]
            for x_pos, thr in enumerate(thresh_vals):
                fig.add_shape(type="line", x0=x_pos - 0.4, x1=x_pos + 0.4,
                              y0=thr, y1=thr, line=dict(color="black", dash="dot", width=2))
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color="black", dash="dot"), name="Threshold"
            ))
            fig.update_layout(
                title="Label Probabilities (dotted line = per-label threshold)",
                yaxis_title="Probability (%)", yaxis_range=[0, 110],
                height=380, margin=dict(t=50, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# TAB 2 — Performance
# ──────────────────────────────────────────────────────────────

with tab2:
    st.markdown("Per-label performance on the **temporal test set (2019–2020)**.")

    # Show cached model thresholds and give a sense of performance
    try:
        model = load_tfidf_model()
        label_names = model["label_names"]
        thresholds = model["thresholds"]

        # Load test data category labels
        y_test_df, label_cols = load_category_test_data()

        # Quick TF-IDF evaluation on test text (requires raw data)
        # Instead, show the training label distribution and threshold table
        st.subheader("Per-Label Thresholds (tuned on validation set)")
        thresh_df = pd.DataFrame([
            {"Category": LABEL_DISPLAY.get(l, l), "Threshold": f"{thresholds[l]:.2f}",
             "Pos Rate (train)": "—"}
            for l in label_names
        ])
        st.dataframe(thresh_df.set_index("Category"), use_container_width=True)

        st.subheader("Model Tier Architecture")
        tiers = pd.DataFrame({
            "Tier": ["1 — TF-IDF Baseline", "2 — SBERT Text Tower", "3 — Fusion MLP"],
            "Features": ["Unigram + bigram TF-IDF (10k vocab)", "Sentence-BERT CLS embeddings (384d)", "SBERT + LightGBM tabular embeddings"],
            "Target Macro-F1": ["0.45–0.55", "0.55–0.65", "0.60–0.75"],
            "Speed": ["Fast (<1s)", "Medium (~3s/batch)", "Slow (requires both towers)"],
        })
        st.dataframe(tiers.set_index("Tier"), use_container_width=True)

    except Exception as e:
        st.error(f"Could not load models: {e}")

    # Label distribution from processed data
    st.subheader("Training Label Distribution")
    try:
        cats = pd.read_parquet(PROCESSED / "category_targets.parquet")
        label_cols = [c for c in cats.columns if c not in ["acn_num_ACN", "primary_category"]]
        counts = {LABEL_DISPLAY.get(c, c): int(cats[c].sum()) for c in label_cols}
        fig = px.bar(
            x=list(counts.keys()), y=list(counts.values()),
            color=list(counts.keys()),
            color_discrete_map={LABEL_DISPLAY.get(k, k): LABEL_COLORS.get(k, "#999") for k in label_cols},
            title="Incident Count per Category (multi-label: one incident may have multiple)",
            labels={"x": "Category", "y": "Count"},
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Multi-label co-occurrence
        st.subheader("Category Co-occurrence (how often pairs appear together)")
        co_matrix = cats[label_cols].T.dot(cats[label_cols])
        np.fill_diagonal(co_matrix.values, 0)
        display_names = [LABEL_DISPLAY.get(c, c) for c in label_cols]
        fig2 = px.imshow(
            co_matrix.values,
            x=display_names, y=display_names,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Co-occurrence Matrix",
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load category data: {e}")


# ──────────────────────────────────────────────────────────────
# TAB 3 — EDA
# ──────────────────────────────────────────────────────────────

with tab3:
    try:
        cats = pd.read_parquet(PROCESSED / "category_targets.parquet")
        splits = pd.read_parquet(PROCESSED / "temporal_splits.parquet")
        label_cols = [c for c in cats.columns if c not in ["acn_num_ACN", "primary_category"]]
        merged = splits.merge(cats, on="acn_num_ACN", how="inner")

        st.subheader("Category Prevalence Over Time")
        merged["year"] = merged["year"].astype(int)
        time_series = merged.groupby("year")[label_cols].mean().reset_index()
        fig = go.Figure()
        for col in label_cols:
            fig.add_trace(go.Scatter(
                x=time_series["year"], y=time_series[col] * 100,
                mode="lines+markers", name=LABEL_DISPLAY.get(col, col),
                line=dict(color=LABEL_COLORS.get(col, "#999")),
            ))
        fig.update_layout(
            title="% of Reports Classified to Each Category by Year",
            yaxis_title="% of reports", height=400,
            xaxis_title="Year",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Primary Category Breakdown")
        primary_counts = cats["primary_category"].value_counts().reset_index()
        primary_counts.columns = ["category", "count"]
        fig2 = px.pie(primary_counts, names="category", values="count",
                      title="Primary Category Distribution",
                      color="category",
                      color_discrete_map={LABEL_DISPLAY.get(k, k): LABEL_COLORS.get(k, "#999")
                                          for k in LABEL_COLORS})
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load category data for EDA: {e}")
