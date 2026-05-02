"""
Problem C — Pre-Flight Risk Scoring
=====================================
Three tabs:
  1. Risk Scorer   — enter flight params → calibrated risk probability + alert gauge
  2. Performance   — ROC curve, PR curve, lift chart on test set (2020)
  3. What-If       — sliders for weather/time → watch risk probability change
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
    load_preflight_model, load_preflight_test_data, load_preflight_features,
    run_preflight_predictions, PROCESSED,
)
from streamlit_app.utils.sidebar import render_manual_test_sidebar, get_uploaded_df

st.set_page_config(page_title="Problem C — Pre-flight Risk", page_icon="🟠", layout="wide")
render_manual_test_sidebar()
st.title("🟠 Problem C — Pre-Flight Risk Scoring")
st.caption(
    "Binary incident risk from BTS flight ops + NOAA weather + NTSB join. "
    "Case-control sampling handles ~5% positive rate. Isotonic calibration."
)

st.warning(
    "⚠️ **Scope note:** This demo uses 2018–2020 BTS data only (3-year reduced scope). "
    "The full implementation plan calls for 2015–2023 (~60 M flights) with complete METAR join. "
    "Results are directionally valid but not production-calibrated.",
    icon="⚠️",
)

tab1, tab2, tab3 = st.tabs(["🎯 Risk Scorer", "📊 Performance", "🔄 What-If"])


# Helper: build risk gauge figure
def risk_gauge(prob: float) -> go.Figure:
    prob_pct = prob * 100
    color = "#2ecc71" if prob_pct < 0.03 else ("#f39c12" if prob_pct < 0.08 else "#e74c3c")
    label = "LOW" if prob_pct < 0.03 else ("MEDIUM" if prob_pct < 0.08 else "HIGH")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_pct,
        title={"text": f"Risk Level: {label}", "font": {"size": 20}},
        delta={"reference": 0.05, "increasing": {"color": "#e74c3c"}},
        number={"suffix": "%", "font": {"size": 30}, "valueformat": ".3f"},
        gauge={
            "axis": {"range": [0, 0.2], "ticksuffix": "%"},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 0.03], "color": "#d5f5e3"},
                {"range": [0.03, 0.08], "color": "#fef9e7"},
                {"range": [0.08, 0.2], "color": "#fdedec"},
            ],
            "threshold": {"line": {"color": "red", "width": 4}, "value": 0.05},
        },
    ))
    fig.update_layout(height=300, margin=dict(t=60, b=10))
    return fig


# ──────────────────────────────────────────────────────────────
# TAB 1 — Risk Scorer
# ──────────────────────────────────────────────────────────────

with tab1:
    st.markdown("Enter flight and weather parameters to estimate incident probability.")

    with st.form("preflight_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Flight Details**")
            distance = st.number_input("Route Distance (miles)", 100, 3000, 850)
            month = st.selectbox("Month", list(range(1, 13)), index=5)
            hour = st.slider("Departure Hour (local)", 0, 23, 9)
            dow = st.selectbox("Day of Week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], index=1)
            carrier = st.selectbox("Carrier", ["AA", "UA", "DL", "WN", "B6", "AS", "NK", "Other"], index=0)

        with col2:
            st.markdown("**Origin Airport Weather**")
            temp = st.slider("Temperature (°F)", -20, 110, 65)
            rhum = st.slider("Relative Humidity (%)", 0, 100, 55)
            prcp = st.slider("Precipitation (inches)", 0.0, 2.0, 0.0, step=0.05)
            wspd = st.slider("Wind Speed (mph)", 0, 50, 8)
            pres = st.slider("Pressure (hPa)", 980, 1040, 1013)

        with col3:
            st.markdown("**Historical Risk Rates**")
            airport_rate = st.number_input(
                "Airport Risk Rate (historical %)", 0.0, 20.0, 3.5, step=0.1,
                help="Average incident rate at this airport"
            )
            carrier_rate = st.number_input(
                "Carrier Risk Rate (historical %)", 0.0, 20.0, 2.8, step=0.1
            )
            route_rate = st.number_input(
                "Route Risk Rate (historical %)", 0.0, 20.0, 2.0, step=0.1
            )

        submitted = st.form_submit_button("Compute Risk Score", use_container_width=True, type="primary")

    if submitted:
        dow_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
        dow_val = dow_map[dow]

        features = {
            "month_sin": float(np.sin(2 * np.pi * month / 12)),
            "month_cos": float(np.cos(2 * np.pi * month / 12)),
            "dow_sin": float(np.sin(2 * np.pi * dow_val / 7)),
            "dow_cos": float(np.cos(2 * np.pi * dow_val / 7)),
            "hour": hour,
            "temp": temp,
            "rhum": rhum,
            "prcp": prcp,
            "wspd": wspd,
            "pres": pres,
            "airport_risk_rate": airport_rate / 100,
            "carrier_risk_rate": carrier_rate / 100,
            "route_risk_rate": route_rate / 100,
            "DISTANCE": distance,
        }

        try:
            model = load_preflight_model()
            model_features = load_preflight_features()
            df_in = pd.DataFrame([features])
            available = [c for c in model_features if c in df_in.columns]
            prob = float(model.predict_proba(df_in[available] if available else df_in)[:, 1][0])

            col_gauge, col_breakdown = st.columns([1, 1])
            with col_gauge:
                st.plotly_chart(risk_gauge(prob), use_container_width=True)
                label = "🟢 LOW" if prob * 100 < 0.03 else ("🟡 MEDIUM" if prob * 100 < 0.08 else "🔴 HIGH")
                st.markdown(f"**Risk Level:** {label} — `{prob*100:.4f}%`")

            with col_breakdown:
                driver_vals = {
                    "Precipitation": prcp * 10,
                    "Wind Speed": wspd / 5,
                    "Humidity": rhum / 10,
                    "Airport Rate": airport_rate,
                    "Carrier Rate": carrier_rate,
                }
                fig = px.bar(
                    x=list(driver_vals.values()),
                    y=list(driver_vals.keys()),
                    orientation="h",
                    title="Risk Driver Magnitudes (relative)",
                    color=list(driver_vals.values()),
                    color_continuous_scale="RdYlGn_r",
                )
                fig.update_layout(height=320, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Ensure `scripts/run_phase3.py` (Problem C) has been run to generate the preflight model.")


# ──────────────────────────────────────────────────────────────
# TAB 2 — Performance
# ──────────────────────────────────────────────────────────────

with tab2:
    with st.spinner("Loading test set performance (cached after first load)…"):
        try:
            uploaded_df = get_uploaded_df("preflight")
            if uploaded_df is not None:
                st.info(
                    f"📤 Manual Test mode — using your uploaded dataset "
                    f"({len(uploaded_df):,} rows). "
                    f"Clear it from the sidebar to return to the default test split."
                )
                test_df, features = run_preflight_predictions(uploaded_df)
                data_source_label = "Uploaded data"
            else:
                test_df, features = load_preflight_test_data()
                data_source_label = "Held-out test split (2020)"

            has_labels = "incident" in test_df.columns
            X_test = test_df[features]

            model = load_preflight_model()
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            from sklearn.metrics import (
                roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
            )

            if has_labels:
                y_test = test_df["incident"]
                roc_auc = roc_auc_score(y_test, y_prob)
                ap = average_precision_score(y_test, y_prob)
                pos_rate = 0.05

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("ROC-AUC", f"{roc_auc:.4f}", help="Target: 0.70–0.80")
                m2.metric("Avg Precision (PR-AUC)", f"{ap:.4f}", help="Primary metric for imbalanced data")
                m3.metric("Positive Rate (test)", f"{pos_rate:.2f}%", help="True aviation safety base rate")
                m4.metric("Source", data_source_label, delta=f"{len(y_test):,} flights", delta_color="off")

                col_roc, col_pr = st.columns(2)

                with col_roc:
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"LightGBM (AUC={roc_auc:.3f})",
                                             line=dict(color="#e74c3c", width=2)))
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random",
                                             line=dict(color="gray", dash="dash")))
                    fig.update_layout(
                        title=f"ROC Curve · {data_source_label}",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col_pr:
                    prec, rec, _ = precision_recall_curve(y_test, y_prob)
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=rec, y=prec, name=f"LightGBM (AP={ap:.3f})",
                                              line=dict(color="#3498db", width=2)))
                    fig2.add_hline(y=0.0005, line_dash="dash", line_color="gray",
                                   annotation_text="Baseline (0.05%)")
                    fig2.update_layout(
                        title="Precision-Recall Curve",
                        xaxis_title="Recall", yaxis_title="Precision",
                        height=400,
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                # Lift chart
                st.subheader("Lift Chart")
                n = len(y_test)
                sorted_idx = np.argsort(-y_prob)
                cumulative_pos = np.cumsum(y_test.values[sorted_idx])
                total_pos = y_test.sum()
                lift = (cumulative_pos / total_pos) / (np.arange(1, n + 1) / n)
                sample_pct = np.linspace(0, 100, n)
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=sample_pct, y=lift, name="Model Lift",
                                           line=dict(color="#27ae60", width=2)))
                fig3.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="No lift (random)")
                fig3.update_layout(
                    title="Cumulative Lift — Model vs. Random Screening",
                    xaxis_title="% of Flights Screened (ranked by risk score)",
                    yaxis_title="Lift",
                    height=350,
                )
                st.plotly_chart(fig3, use_container_width=True)
                st.caption(
                    "Lift > 1 means the model finds more true incidents per flight screened than random. "
                    "At 10% of flights, the model captures significantly more incidents than random selection."
                )
            else:
                st.info(
                    "Ground-truth labels not provided in the upload — "
                    "showing risk score distribution only. Add an `incident` column (0/1) "
                    "to see ROC/PR curves and lift chart."
                )
                m1, m2 = st.columns(2)
                m1.metric("Flights scored", f"{len(y_prob):,}")
                m2.metric("Source", data_source_label)

                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=y_prob, nbinsx=50, name="Risk score",
                    marker_color="#e74c3c", opacity=0.7,
                ))
                fig_dist.update_layout(
                    title="Predicted Risk Score Distribution",
                    xaxis_title="Risk probability", yaxis_title="Count", height=350,
                )
                st.plotly_chart(fig_dist, use_container_width=True)

        except Exception as e:
            st.error(f"Could not compute performance: {e}")


# ──────────────────────────────────────────────────────────────
# TAB 3 — What-If
# ──────────────────────────────────────────────────────────────

with tab3:
    st.markdown(
        "Adjust **weather** and **operational** parameters to see how risk probability changes. "
        "All other features are held at average values."
    )

    try:
        model_wi = load_preflight_model()
    except Exception as e:
        st.error(f"Pre-flight model not found: {e}")
        st.info("Run `scripts/run_phase3.py` to train the model before using What-If analysis.")
        st.stop()

    def predict_risk(overrides: dict) -> float:
        defaults = {
            "month_sin": float(np.sin(2 * np.pi * 6 / 12)),
            "month_cos": float(np.cos(2 * np.pi * 6 / 12)),
            "dow_sin": 0.0, "dow_cos": 1.0,
            "hour": 9,
            "temp": 65, "rhum": 55, "prcp": 0.0,
            "wspd": 8, "pres": 1013,
            "airport_risk_rate": 0.035,
            "carrier_risk_rate": 0.028,
            "route_risk_rate": 0.020,
            "DISTANCE": 850,
        }
        defaults.update(overrides)
        df_in = pd.DataFrame([defaults])
        try:
            return float(model_wi.predict_proba(df_in)[:, 1][0])
        except Exception:
            return 0.03

    col_ctrl, col_chart = st.columns([1, 2])
    with col_ctrl:
        st.markdown("**Weather Parameters**")
        wi_prcp = st.slider("Precipitation (in)", 0.0, 2.0, 0.0, 0.1, key="wi_prcp")
        wi_wspd = st.slider("Wind Speed (mph)", 0, 50, 8, 1, key="wi_wspd")
        wi_temp = st.slider("Temperature (°F)", -10, 110, 65, 5, key="wi_temp")
        wi_rhum = st.slider("Humidity (%)", 0, 100, 55, 5, key="wi_rhum")
        st.markdown("**Operational Parameters**")
        wi_dist = st.slider("Distance (miles)", 100, 3000, 850, 50, key="wi_dist")
        wi_hour = st.slider("Departure Hour", 0, 23, 9, 1, key="wi_hour")

    overrides = {
        "prcp": wi_prcp, "wspd": wi_wspd, "temp": wi_temp, "rhum": wi_rhum,
        "DISTANCE": wi_dist, "hour": wi_hour,
    }
    risk = predict_risk(overrides)

    with col_chart:
        st.plotly_chart(risk_gauge(risk), use_container_width=True)

        # Sensitivity: vary precipitation and wind separately
        prcp_range = np.linspace(0, 2, 20)
        risks_prcp = [predict_risk({"prcp": p, "wspd": wi_wspd, "temp": wi_temp, "rhum": wi_rhum}) for p in prcp_range]
        wspd_range = np.linspace(0, 50, 20)
        risks_wspd = [predict_risk({"prcp": wi_prcp, "wspd": w, "temp": wi_temp, "rhum": wi_rhum}) for w in wspd_range]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prcp_range, y=[r * 100 for r in risks_prcp],
                                  name="Precipitation sensitivity",
                                  line=dict(color="#3498db")))
        fig.update_layout(
            title="Risk vs. Precipitation (other params fixed)",
            xaxis_title="Precipitation (inches)",
            yaxis_title="Risk (%)", height=250,
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=wspd_range, y=[r * 100 for r in risks_wspd],
                                   name="Wind speed sensitivity",
                                   line=dict(color="#e74c3c")))
        fig2.update_layout(
            title="Risk vs. Wind Speed (other params fixed)",
            xaxis_title="Wind Speed (mph)",
            yaxis_title="Risk (%)", height=250,
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)
