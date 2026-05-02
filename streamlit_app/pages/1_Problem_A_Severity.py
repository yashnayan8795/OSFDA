"""
Problem A — Incident Severity Triage
======================================
Three tabs:
  1. Live Prediction  — form → severity level + probability bars + feature importance
  2. Performance      — confusion matrix, QWK, calibration, per-class metrics
  3. What-If          — adjust top features with sliders, watch prediction change
"""

import re
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
    load_severity_model, load_severity_test_predictions, run_severity_predictions,
)
from streamlit_app.utils.sidebar import render_manual_test_sidebar, get_uploaded_df

st.set_page_config(page_title="Problem A — Severity", page_icon="🔴", layout="wide")
render_manual_test_sidebar()
st.title("🔴 Problem A — Incident Severity Triage")
st.caption(
    "Ordinal severity 0–3 from pre-circumstance features only. "
    "LightGBM + CatBoost with isotonic calibration and QWK metric."
)

SEVERITY_LABELS = {0: "Minor", 1: "Moderate", 2: "Substantial", 3: "Critical"}
SEVERITY_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e67e22", 3: "#e74c3c"}

tab1, tab2, tab3 = st.tabs(["🎯 Live Prediction", "📊 Performance Dashboard", "🔄 What-If Analysis"])


# ──────────────────────────────────────────────────────────────
# Helpers — cached once per session
# ──────────────────────────────────────────────────────────────

@st.cache_resource
def _get_cat_feature_names():
    """Parse which feature names the LightGBM model treats as categorical."""
    model_path = ROOT / "models" / "severity_lgbm.txt"
    if not model_path.exists():
        return set()
    try:
        model, _ = load_severity_model()
        feature_names = model.feature_name()
        match = re.search(
            r'\[categorical_feature: ([0-9,]+)\]',
            model_path.read_text(encoding="utf-8"),
        )
        if match:
            return {
                feature_names[int(i)]
                for i in match.group(1).split(",")
                if int(i) < len(feature_names)
            }
    except Exception:
        pass
    return set()


def _cast_for_lgbm(df: pd.DataFrame, cat_names: set) -> pd.DataFrame:
    """Cast columns the model expects as categorical to category dtype."""
    for col in df.columns:
        if col in cat_names:
            val = df[col].iloc[0]
            if val is not None and not isinstance(val, str):
                df[col] = df[col].astype(str)
            df[col] = df[col].astype("category")
    return df


def _predict(model, calibrators, df_in):
    """Run model + calibrators, return (predicted_class, cal_probs)."""
    raw_probs = model.predict(df_in)
    cal = np.array([c.predict([raw_probs[0, i]])[0] for i, c in enumerate(calibrators)])
    cal = cal / cal.sum()
    return int(cal.argmax()), cal


# ──────────────────────────────────────────────────────────────
# TAB 1 — Live Prediction
# ──────────────────────────────────────────────────────────────

with tab1:
    st.markdown("Enter incident context features to get a severity prediction.")
    st.info(
        "These are **pre-circumstance** features only — no post-incident observations. "
        "The leakage audit excluded all `Events.*` and `Assessments.*` columns.",
        icon="🛡️",
    )

    with st.form("severity_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Environment**")
            flight_conditions = st.selectbox(
                "Flight Conditions", ["VMC", "IMC", "Mixed", "Marginal", ""], index=0
            )
            light = st.selectbox("Light", ["Daylight", "Night", "Dawn", "Dusk", ""], index=0)
            ceiling = st.text_input("Ceiling (ft, blank = clear)", value="")
            rvr = st.text_input("RVR (ft, blank = N/A)", value="")
            weather = st.selectbox(
                "Weather Elements",
                ["", "None", "Rain", "Snow", "Fog; Rain; 0.5", "Turbulence", "Icing; Snow; 0.5", "Clear"],
                index=1,
            )

        with col2:
            st.markdown("**Aircraft & Operations**")
            far_part = st.selectbox(
                "FAR Part", ["Part 121", "Part 91", "Part 135", "Part 129", ""], index=0
            )
            flight_plan = st.selectbox("Flight Plan", ["IFR", "VFR", "DVFR", ""], index=0)
            mission = st.selectbox(
                "Mission",
                ["", "Air Carrier", "Personal", "Training", "Cargo / Air Freight", "Other"],
                index=1,
            )
            work_env = st.selectbox(
                "Work Environment",
                ["", "None Identified", "Poor Lighting", "Temperature - Extreme", "Excessive Noise"],
                index=1,
            )

        with col3:
            st.markdown("**Crew & Context**")
            function = st.selectbox(
                "Reporter Function",
                ["Captain", "First Officer", "Flight Attendant", "Air Traffic Control", "Mechanic", ""],
                index=0,
            )
            qualification = st.selectbox(
                "Qualification", ["ATP", "Commercial", "Instrument", "Student", ""], index=0
            )
            experience = st.number_input(
                "Total Flight Hours", min_value=0, max_value=50000, value=5000, step=500
            )
            state = st.selectbox(
                "State", ["CA", "TX", "FL", "NY", "IL", "CO", "WA", "GA", ""], index=0
            )
            year = st.number_input("Report Year", min_value=2003, max_value=2020, value=2018)
            month = st.selectbox("Month", list(range(1, 13)), index=5)

        submitted = st.form_submit_button("Predict Severity", use_container_width=True, type="primary")

    if submitted:
        try:
            model, calibrators = load_severity_model()
            feature_names = model.feature_name()
            cat_names = _get_cat_feature_names()

            month_sin = float(np.sin(2 * np.pi * month / 12))
            month_cos = float(np.cos(2 * np.pi * month / 12))
            quarter = (month - 1) // 3 + 1
            time_bucket = "1201-1800"

            input_map = {
                "Environment_Flight_Conditions": flight_conditions or None,
                "Environment.3_Light": light or None,
                "Environment.4_Ceiling": ceiling or None,
                "Environment.5_RVR.Single_Value": rvr or None,
                "Environment.1_Weather_Elements_/_Visibility": weather or None,
                "Environment.2_Work_Environment_Factor": work_env or None,
                "Aircraft_1.5_Operating_Under_FAR_Part": far_part or None,
                "Aircraft_1.6_Flight_Plan": flight_plan or None,
                "Aircraft_1.7_Mission": mission or None,
                "Person_1.3_Function": function or None,
                "Person_1.4_Qualification": qualification or None,
                "Person_1.5_Experience": str(experience) if experience > 0 else None,
                "Place.1_State_Reference": state or None,
                "time_of_day_bucket": time_bucket,
                "Time.1_Local_Time_Of_Day": time_bucket,
                "year": year,
                "month": month,
                "quarter": quarter,
                "month_sin": month_sin,
                "month_cos": month_cos,
                "Place_Locale_Reference": None,
            }

            row = {f: [input_map.get(f)] for f in feature_names}
            df_in = _cast_for_lgbm(pd.DataFrame(row), cat_names)
            predicted_class, cal_probs = _predict(model, calibrators, df_in)

            col_pred, col_probs = st.columns([1, 2])
            with col_pred:
                label = SEVERITY_LABELS[predicted_class]
                color = SEVERITY_COLORS[predicted_class]
                st.markdown(
                    f"""
                    <div style="background:{color};padding:20px;border-radius:12px;text-align:center;">
                        <h1 style="color:white;margin:0">Level {predicted_class}</h1>
                        <h2 style="color:white;margin:0">{label}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("")
                st.metric("Confidence", f"{cal_probs[predicted_class]*100:.1f}%")

            with col_probs:
                fig = go.Figure(go.Bar(
                    x=[f"Level {i}<br>{SEVERITY_LABELS[i]}" for i in range(4)],
                    y=cal_probs * 100,
                    marker_color=[SEVERITY_COLORS[i] for i in range(4)],
                    text=[f"{p:.1f}%" for p in cal_probs * 100],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="Predicted Probability per Class",
                    yaxis_title="Probability (%)",
                    yaxis_range=[0, 105],
                    height=350,
                    margin=dict(t=40, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

            importance = model.feature_importance(importance_type="gain")
            names = model.feature_name()
            top_idx = np.argsort(importance)[::-1][:12]
            fig2 = px.bar(
                x=importance[top_idx],
                y=[names[i].replace("_", " ") for i in top_idx],
                orientation="h",
                title="Top Feature Importances (gain)",
                labels={"x": "Importance (gain)", "y": "Feature"},
                color=importance[top_idx],
                color_continuous_scale="Blues",
            )
            fig2.update_layout(height=380, margin=dict(t=40, b=10), showlegend=False)
            fig2.update_yaxes(autorange="reversed")
            st.plotly_chart(fig2, use_container_width=True)

        except FileNotFoundError:
            st.error("Model file not found. Run `scripts/run_phase2.py` first to train the severity model.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ──────────────────────────────────────────────────────────────
# TAB 2 — Performance Dashboard
# ──────────────────────────────────────────────────────────────

with tab2:
    with st.spinner("Loading test-set predictions (first time only)…"):
        try:
            uploaded_df = get_uploaded_df("severity")
            if uploaded_df is not None:
                st.info(
                    f"📤 Manual Test mode — using your uploaded dataset "
                    f"({len(uploaded_df):,} rows). "
                    f"Clear it from the sidebar to return to the default test split."
                )
                y_true, y_pred, cal_probs, X_test, feature_cols = run_severity_predictions(uploaded_df)
                data_source_label = "Uploaded data"
            else:
                y_true, y_pred, cal_probs, X_test, feature_cols = load_severity_test_predictions()
                data_source_label = "Held-out test split (2020)"

            has_labels = np.any(y_true >= 0)

            from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

            if has_labels:
                qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
                mae = float(np.mean(np.abs(y_true - y_pred)))
                cm = confusion_matrix(y_true, y_pred)
                cls_report = classification_report(y_true, y_pred, output_dict=True)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("QWK (Primary)", f"{qwk:.4f}", help="Quadratic Weighted Kappa — target ≥ 0.45")
                m2.metric("Ordinal MAE", f"{mae:.3f}", help="Average absolute ordinal distance")
                m3.metric("Macro-F1", f"{cls_report['macro avg']['f1-score']:.3f}")
                m4.metric("Source", data_source_label, delta=f"{len(y_true):,} rows", delta_color="off")

                col_cm, col_cls = st.columns(2)
                with col_cm:
                    fig = px.imshow(
                        cm,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=[f"Pred {SEVERITY_LABELS[i]}" for i in range(4)],
                        y=[f"True {SEVERITY_LABELS[i]}" for i in range(4)],
                        color_continuous_scale="Blues",
                        text_auto=True,
                        title="Confusion Matrix",
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                with col_cls:
                    rows = []
                    for i in range(4):
                        r = cls_report.get(str(i), {})
                        rows.append({
                            "Level": f"{i} — {SEVERITY_LABELS[i]}",
                            "Precision": round(r.get("precision", 0), 3),
                            "Recall": round(r.get("recall", 0), 3),
                            "F1": round(r.get("f1-score", 0), 3),
                            "Support": int(r.get("support", 0)),
                        })
                    st.dataframe(
                        pd.DataFrame(rows).set_index("Level"),
                        use_container_width=True,
                        height=200,
                    )
                    st.caption("QWK penalises large ordinal jumps (e.g. 0→3) more than adjacent errors (0→1).")
            else:
                st.info(
                    "Ground-truth labels not provided in the upload — "
                    "showing prediction distribution only. Add a `Severity_Label` "
                    "column (0–3) to see evaluation metrics."
                )
                m1, m2 = st.columns(2)
                m1.metric("Predictions", f"{len(y_pred):,}")
                m2.metric("Source", data_source_label)

            st.subheader("Severity Distribution")
            fig3 = go.Figure()
            if has_labels:
                fig3.add_trace(go.Histogram(x=y_true, name="Actual", opacity=0.7, marker_color="coral"))
            fig3.add_trace(go.Histogram(x=y_pred, name="Predicted", opacity=0.7, marker_color="steelblue"))
            fig3.update_layout(
                barmode="overlay",
                title=f"Predicted vs Actual Distribution · {data_source_label}",
                xaxis_title="Severity Level",
                yaxis_title="Count",
                height=300,
            )
            st.plotly_chart(fig3, use_container_width=True)

        except FileNotFoundError:
            st.error("Model artifacts not found.")
            st.info("Run `scripts/run_phase2.py` first to generate model artifacts.")
        except Exception as e:
            st.error(f"Could not load test predictions: {e}")
            st.info("Run `scripts/run_phase2.py` first to generate model artifacts.")


# ──────────────────────────────────────────────────────────────
# TAB 3 — What-If Analysis
# ──────────────────────────────────────────────────────────────

with tab3:
    st.markdown(
        "Adjust single features below to see how the severity prediction shifts. "
        "All other inputs hold constant at their default values."
    )

    try:
        model, calibrators = load_severity_model()
        feature_names = model.feature_name()
        cat_names = _get_cat_feature_names()

        def base_prediction(overrides: dict) -> np.ndarray:
            defaults = {
                "Environment_Flight_Conditions": "VMC",
                "Environment.3_Light": "Daylight",
                "Environment.4_Ceiling": None,
                "Environment.5_RVR.Single_Value": None,
                "Environment.1_Weather_Elements_/_Visibility": "None",
                "Environment.2_Work_Environment_Factor": "None Identified",
                "Aircraft_1.5_Operating_Under_FAR_Part": "Part 121",
                "Aircraft_1.6_Flight_Plan": "IFR",
                "Aircraft_1.7_Mission": "Air Carrier",
                "Person_1.3_Function": "Captain",
                "Person_1.4_Qualification": "ATP",
                "Person_1.5_Experience": "8000",
                "Place.1_State_Reference": "CA",
                "time_of_day_bucket": "1201-1800",
                "Time.1_Local_Time_Of_Day": "1201-1800",
                "year": 2019,
                "month": 6,
                "quarter": 2,
                "month_sin": float(np.sin(2 * np.pi * 6 / 12)),
                "month_cos": float(np.cos(2 * np.pi * 6 / 12)),
                "Place_Locale_Reference": None,
            }
            defaults.update(overrides)
            row = {f: [defaults.get(f)] for f in feature_names}
            df_in = _cast_for_lgbm(pd.DataFrame(row), cat_names)
            try:
                _, cal = _predict(model, calibrators, df_in)
                return cal
            except Exception:
                return np.array([0.25, 0.25, 0.25, 0.25])

        col_sliders, col_output = st.columns([1, 2])

        with col_sliders:
            st.markdown("**Vary these features:**")
            flight_cond_wi = st.selectbox(
                "Flight Conditions", ["VMC", "IMC", "Mixed", "Marginal"], key="wi_fc"
            )
            light_wi = st.selectbox("Light", ["Daylight", "Night", "Dawn", "Dusk"], key="wi_light")
            far_wi = st.selectbox(
                "FAR Part", ["Part 121", "Part 91", "Part 135"], key="wi_far"
            )
            fn_wi = st.selectbox(
                "Reporter Function",
                ["Captain", "First Officer", "Flight Attendant", "Air Traffic Control"],
                key="wi_fn",
            )
            exp_wi = st.slider("Flight Hours", 0, 30000, 5000, 500, key="wi_exp")

        overrides = {
            "Environment_Flight_Conditions": flight_cond_wi,
            "Environment.3_Light": light_wi,
            "Aircraft_1.5_Operating_Under_FAR_Part": far_wi,
            "Person_1.3_Function": fn_wi,
            "Person_1.5_Experience": str(exp_wi),
        }
        probs = base_prediction(overrides)
        pred = int(probs.argmax())

        with col_output:
            fig = go.Figure()
            categories = [f"Level {i}<br>{SEVERITY_LABELS[i]}" for i in range(4)]
            fig.add_trace(go.Bar(
                x=categories, y=probs * 100,
                marker_color=[SEVERITY_COLORS[i] for i in range(4)],
                text=[f"{p:.1f}%" for p in probs * 100],
                textposition="outside",
            ))
            fig.update_layout(
                title=f"Current Prediction: Level {pred} — {SEVERITY_LABELS[pred]}",
                yaxis_range=[0, 105], height=380, margin=dict(t=50, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"**Confidence:** `{probs[pred]*100:.1f}%`")
            st.caption(
                "Tip: switch Flight Conditions to IMC + Light to Night to see severity shift upward."
            )

    except FileNotFoundError:
        st.error("Severity model not found.")
        st.info("Run `scripts/run_phase2.py` to train the model before using What-If analysis.")
    except Exception as e:
        st.error(f"What-If analysis unavailable: {e}")
