import pandas as pd
import numpy as np
from app.services.model_loader import ModelStore
from src.features.temporal import extract_temporal_features
from src.features.encoding import bucket_experience, prepare_for_lgbm, identify_column_types
from src.models.severity import predict_calibrated

SEVERITY_LABELS = ["Minor", "Moderate", "Substantial", "Critical"]

def _build_row(fields: dict) -> pd.DataFrame:
    # Map API fields to ASRS column names used in training
    row = {
        "Time_Date": fields.get("time_date"),
        "Time.1_Local Time Of Day": fields.get("time_of_day"),
        "Place_Locale Reference": fields.get("locale_reference"),
        "Place.1_State Reference": fields.get("state_reference"),
        "Environment_Flight Conditions": fields.get("flight_conditions"),
        "Environment.1_Weather Elements / Visibility": fields.get("weather_elements"),
        "Environment.2_Work Environment Factor": fields.get("work_env_factor"),
        "Environment.3_Light": fields.get("light"),
        "Environment.4_Ceiling": fields.get("ceiling"),
        "Environment.5_RVR.Single Value": fields.get("rvr"),
        "Aircraft 1_Make": fields.get("aircraft_make"),
        "Aircraft 1.1_Model": fields.get("aircraft_model"),
        "Aircraft 1.2_Operator": fields.get("operator"),
        "Aircraft 1.5_Operating Under FAR Part": fields.get("far_part"),
        "Aircraft 1.6_Flight Plan": fields.get("flight_plan"),
        "Aircraft 1.7_Mission": fields.get("mission"),
        "Aircraft 1.8_Flight Phase": fields.get("flight_phase"),
        "Aircraft 1.14_Number of Seats": fields.get("num_seats"),
        "Aircraft 1.15_Crew Size": fields.get("crew_size"),
        "Person 1.3_Function": fields.get("person_function"),
        "Person 1.4_Qualification": fields.get("qualification"),
        "Person 1.5_Experience": fields.get("experience_hours"),
    }
    return pd.DataFrame([row])

def classify_severity(fields: dict) -> dict:
    df = _build_row(fields)

    # Feature Engineering (reuse src modules)
    from src.data.loader import parse_time_date
    df = parse_time_date(df)
    df = extract_temporal_features(df)
    df = bucket_experience(df)

    # Align with model features and ensure types
    feature_cols = ModelStore.severity_feature_cols or []
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
    
    # Identify types from the aligned dataframe
    col_types = identify_column_types(df[feature_cols])
    
    # Prepare X
    X = df[feature_cols].copy()
    
    # Ensure categorical columns are strings (CatBoost requirement)
    for col in col_types["categorical"]:
        # If it's a known numeric col that's currently 'object' due to None, force it to numeric
        if any(x in col.lower() for x in ["experience", "rvr", "seats", "size", "year", "month", "sin", "cos", "qwk", "quarter"]):
            X[col] = pd.to_numeric(X[col], errors='coerce').astype(float)
        else:
            X[col] = X[col].astype(str).replace("nan", "Missing").replace("None", "Missing").fillna("Missing")
    
    # Ensure numeric columns are floats
    for col in col_types["numeric"]:
        X[col] = pd.to_numeric(X[col], errors='coerce').astype(float)

    print(f"[Severity] X dtypes:\n{X.dtypes}")
    print(f"[Severity] X row:\n{X.iloc[0].to_dict()}")

    model = ModelStore.severity_model
    calibrators = ModelStore.severity_calibrators

    if model is None:
        return {
            "level": 0, "label": "Model Not Loaded", 
            "probabilities": [0.25]*4, "confidence": 0.0,
            "cost_sensitive_recommendation": "Check system health"
        }

    # Prediction
    if calibrators:
        probs = predict_calibrated(model, X, calibrators)
    else:
        probs = model.predict_proba(X)

    level = int(probs[0].argmax())
    
    # Cost-sensitive logic (simplified)
    # Higher levels or low confidence in critical levels trigger "Safety Triage"
    recommendation = "Standard Review"
    if level >= 2: # Substantial or Critical
        recommendation = "IMMEDIATE SAFETY AUDIT"
    elif level == 1 and probs[0][1] > 0.6:
        recommendation = "Priority Investigation"
    elif probs[0][2:].sum() > 0.3: # Significant tail risk
        recommendation = "Enhanced Monitoring"

    return {
        "level": level,
        "label": SEVERITY_LABELS[level],
        "probabilities": [float(p) for p in probs[0]],
        "confidence": float(probs[0][level]),
        "cost_sensitive_recommendation": recommendation
    }

def get_severity_feature_schema():
    # In a real app, this would be derived from severity_feature_cols.json
    # and a mapping of column names to display names/types.
    cols = ModelStore.severity_feature_cols or []
    schema = []
    for col in cols:
        schema.append({
            "name": col,
            "display_name": col.replace("_", " ").title(),
            "type": "categorical" if "bucket" in col or col in ["month", "quarter", "Time.1_Local Time Of Day"] else "numeric",
            "allowed_values": None,
            "required": False,
            "description": f"ASRS feature: {col}"
        })
    return schema

def get_severity_distribution():
    # Return mock or pre-calculated stats
    return {
        "minor": 15000, "moderate": 10000, "substantial": 3000, "critical": 655, "total": 38655
    }
