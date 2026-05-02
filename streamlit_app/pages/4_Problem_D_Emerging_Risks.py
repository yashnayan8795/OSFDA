"""
Problem D — Emerging Risk Discovery
=====================================
Three tabs:
  1. Risk Ranking    — top emerging topics ranked by risk score, with growth/severity breakdown
  2. Trend Explorer  — topic count over time with PELT changepoint markers
  3. Topic Browser   — pick a topic, see keywords, trend, and severity trajectory
"""

import sys
from pathlib import Path
import ast
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from streamlit_app.utils.loaders import load_emerging_risks
from streamlit_app.utils.sidebar import render_manual_test_sidebar

st.set_page_config(page_title="Problem D — Emerging Risks", page_icon="🟣", layout="wide")
render_manual_test_sidebar()
st.title("🟣 Problem D — Emerging Risk Discovery")
st.caption(
    "Unsupervised narrative clustering (BERTopic / PCA+MiniBatchKMeans) + "
    "PELT changepoint detection. Severity-weighted risk scoring surfaces rising themes."
)

st.warning(
    "⚠️ **Implementation note:** UMAP+HDBSCAN was planned but blocked by Windows Numba DLL "
    "policy. PCA (15 components) + MiniBatchKMeans was substituted. Topic coherence is lower "
    "than the full spec but trend detection is unaffected.",
    icon="⚠️",
)

try:
    risks, trends, reps, changepoints = load_emerging_risks()
except Exception as e:
    st.error(f"Could not load emerging risk data: {e}")
    st.info("Run `scripts/run_phase4.py` to generate the required artefacts.")
    st.stop()

# Parse Representation column if stored as string
def _get_keywords(topic_id):
    key = str(int(topic_id))
    val = reps.get(key, reps.get(int(topic_id), {}))
    if isinstance(val, dict):
        return val.get("keywords", [])
    if isinstance(val, list):
        return val
    return []


def _topic_label(row) -> str:
    name = str(row.get("Name", row.get("Topic", "")))
    parts = name.split("_", 1)
    return parts[1].replace("_", " ").title() if len(parts) > 1 else name


risks["label"] = risks.apply(_topic_label, axis=1)
risks_sorted = risks.sort_values("Risk_Score", ascending=False).reset_index(drop=True)

tab1, tab2, tab3 = st.tabs(["🏆 Risk Ranking", "📈 Trend Explorer", "🔍 Topic Browser"])


# ──────────────────────────────────────────────────────────────
# TAB 1 — Risk Ranking
# ──────────────────────────────────────────────────────────────

with tab1:
    st.markdown(
        "Topics ranked by **Risk Score** = `Growth_Ratio × Avg_Severity × log(Count)`. "
        "A recent PELT changepoint boosts the score."
    )

    top_n = st.slider("Show top N topics", 5, 30, 15, key="top_n_rank")
    df_top = risks_sorted.head(top_n).copy()

    # Color by whether a recent changepoint was detected
    df_top["Changepoint"] = df_top["Recent_Changepoint"].map({True: "Yes", False: "No"})

    fig = px.bar(
        df_top,
        x="Risk_Score",
        y="label",
        orientation="h",
        color="Changepoint",
        color_discrete_map={"Yes": "#e74c3c", "No": "#3498db"},
        hover_data={"Growth_Ratio": ":.2f", "Avg_Severity": ":.2f", "Count": True},
        title=f"Top {top_n} Emerging Risk Topics by Risk Score",
        labels={"Risk_Score": "Risk Score", "label": "Topic"},
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=max(350, top_n * 28), margin=dict(l=200, t=50, b=30))
    st.plotly_chart(fig, use_container_width=True)

    # Scatter: Growth Ratio vs Avg Severity
    st.subheader("Growth vs. Severity Landscape")
    fig2 = px.scatter(
        risks_sorted,
        x="Growth_Ratio",
        y="Avg_Severity",
        size="Count",
        color="Risk_Score",
        color_continuous_scale="RdYlGn_r",
        hover_name="label",
        hover_data={"Risk_Score": ":.2f", "Count": True, "Recent_Changepoint": True},
        title="Topic Landscape: Growth Ratio × Avg Severity (bubble size = report count)",
        labels={"Growth_Ratio": "Growth Ratio (recent / baseline)", "Avg_Severity": "Avg Severity"},
    )
    fig2.add_vline(x=1.0, line_dash="dash", line_color="gray", annotation_text="No growth")
    fig2.update_layout(height=450)
    st.plotly_chart(fig2, use_container_width=True)

    # Summary table
    st.subheader("Full Risk Table")
    display_df = risks_sorted[["label", "Risk_Score", "Growth_Ratio", "Avg_Severity", "Count", "Recent_Changepoint"]].copy()
    display_df.columns = ["Topic", "Risk Score", "Growth Ratio", "Avg Severity", "Reports", "Recent Changepoint"]
    display_df["Risk Score"] = display_df["Risk Score"].round(3)
    display_df["Growth Ratio"] = display_df["Growth Ratio"].round(3)
    display_df["Avg Severity"] = display_df["Avg Severity"].round(3)
    st.dataframe(display_df.set_index("Topic"), use_container_width=True)


# ──────────────────────────────────────────────────────────────
# TAB 2 — Trend Explorer
# ──────────────────────────────────────────────────────────────

with tab2:
    st.markdown(
        "Select topics to overlay their **report count trends**. "
        "Red vertical lines mark PELT-detected changepoints."
    )

    all_labels = risks_sorted["label"].tolist()
    topic_ids = risks_sorted["Topic"].tolist()
    label_to_id = dict(zip(all_labels, topic_ids))

    default_sel = all_labels[:5]
    selected = st.multiselect(
        "Topics to display (ranked by risk score)",
        options=all_labels,
        default=default_sel,
        key="trend_sel",
    )

    if not selected:
        st.info("Select at least one topic above.")
    else:
        # Normalize toggle
        normalize = st.checkbox("Normalize counts (relative to each topic's max)", value=False)

        fig = go.Figure()
        colors = px.colors.qualitative.Plotly
        periods_all = sorted(trends["period"].unique())
        period_idx = {p: i for i, p in enumerate(periods_all)}

        for idx, label in enumerate(selected):
            tid = label_to_id[label]
            topic_data = trends[trends["topic_id"] == tid].sort_values("period")
            if topic_data.empty:
                continue

            y_vals = topic_data["count"].values.astype(float)
            if normalize and y_vals.max() > 0:
                y_vals = y_vals / y_vals.max()

            color = colors[idx % len(colors)]
            fig.add_trace(go.Scatter(
                x=topic_data["period"],
                y=y_vals,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=5),
            ))

            # Changepoint markers — use add_shape+add_annotation; add_vline with
            # annotation_text crashes on categorical x-axes (plotly tries to mean strings)
            cp_indices = changepoints.get(str(int(tid)), changepoints.get(int(tid), []))
            topic_periods = topic_data["period"].tolist()
            for cp_i in cp_indices:
                if cp_i < len(topic_periods):
                    cp_period = topic_periods[cp_i]
                    fig.add_shape(
                        type="line", x0=cp_period, x1=cp_period,
                        y0=0, y1=1, yref="paper",
                        line=dict(color=color, width=1, dash="dot"),
                    )
                    fig.add_annotation(
                        x=cp_period, y=1.04, yref="paper",
                        text="CP", font=dict(color=color, size=9),
                        showarrow=False, xanchor="center",
                    )

        y_label = "Normalized Count" if normalize else "Report Count"
        fig.update_layout(
            title="Topic Report Volume Over Time",
            xaxis_title="Period",
            yaxis_title=y_label,
            height=450,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Severity trajectory for selected topics
        st.subheader("Severity Trajectory")
        fig2 = go.Figure()
        for idx, label in enumerate(selected):
            tid = label_to_id[label]
            topic_data = trends[trends["topic_id"] == tid].sort_values("period")
            if topic_data.empty or "avg_severity" not in topic_data.columns:
                continue
            fig2.add_trace(go.Scatter(
                x=topic_data["period"],
                y=topic_data["avg_severity"],
                mode="lines+markers",
                name=label,
                line=dict(color=colors[idx % len(colors)], width=2),
                marker=dict(size=5),
            ))
        fig2.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Avg severity = 1")
        fig2.update_layout(
            title="Average Incident Severity by Topic Over Time",
            xaxis_title="Period",
            yaxis_title="Avg Severity (0 = none, 3 = fatal)",
            height=350,
            hovermode="x unified",
        )
        st.plotly_chart(fig2, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# TAB 3 — Topic Browser
# ──────────────────────────────────────────────────────────────

with tab3:
    st.markdown("Deep-dive into a single topic: keywords, trend, and severity profile.")

    selected_label = st.selectbox(
        "Select topic",
        options=all_labels,
        index=0,
        key="browser_topic",
    )
    tid = label_to_id[selected_label]
    row = risks_sorted[risks_sorted["Topic"] == tid].iloc[0]
    keywords = _get_keywords(tid)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Score", f"{row['Risk_Score']:.3f}")
    c2.metric("Growth Ratio", f"{row['Growth_Ratio']:.3f}", help="Recent period vs. baseline")
    c3.metric("Avg Severity", f"{row['Avg_Severity']:.3f}", help="0=none, 3=fatal")
    c4.metric("Reports", f"{int(row['Count']):,}")

    if row["Recent_Changepoint"]:
        st.error("⚠️ PELT changepoint detected in recent periods — rapid acceleration of this theme.")
    else:
        st.success("No recent changepoint — trend is stable or gradually rising.")

    # Keywords
    if keywords:
        st.subheader("Topic Keywords")
        keyword_html = " ".join(
            f'<span style="background:#2c3e50;color:white;padding:3px 10px;'
            f'border-radius:10px;margin:3px;font-size:0.9em;">{k}</span>'
            for k in keywords[:12]
        )
        st.markdown(keyword_html, unsafe_allow_html=True)
    else:
        st.info("No keyword representation available for this topic.")

    # Trend chart
    topic_data = trends[trends["topic_id"] == tid].sort_values("period")
    if not topic_data.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=topic_data["period"],
            y=topic_data["count"],
            name="Report Count",
            marker_color="#3498db",
        ))

        cp_indices = changepoints.get(str(int(tid)), changepoints.get(int(tid), []))
        topic_periods = topic_data["period"].tolist()
        for cp_i in cp_indices:
            if cp_i < len(topic_periods):
                fig.add_shape(
                    type="line", x0=topic_periods[cp_i], x1=topic_periods[cp_i],
                    y0=0, y1=1, yref="paper",
                    line=dict(color="#e74c3c", width=3),
                )
                fig.add_annotation(
                    x=topic_periods[cp_i], y=1.06, yref="paper",
                    text="Changepoint", font=dict(color="#e74c3c", size=11),
                    showarrow=False, xanchor="right",
                )

        fig.update_layout(
            title=f"Report Volume — {selected_label}",
            xaxis_title="Period",
            yaxis_title="Report Count",
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

        if "avg_severity" in topic_data.columns:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=topic_data["period"],
                y=topic_data["avg_severity"],
                mode="lines+markers",
                fill="tozeroy",
                fillcolor="rgba(231,76,60,0.15)",
                line=dict(color="#e74c3c", width=2),
                marker=dict(size=6),
                name="Avg Severity",
            ))
            fig2.add_hline(y=1.0, line_dash="dash", line_color="gray")
            fig2.update_layout(
                title=f"Severity Trend — {selected_label}",
                xaxis_title="Period",
                yaxis_title="Avg Severity",
                height=270,
                yaxis=dict(range=[0, 3.2]),
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No trend data available for this topic.")
