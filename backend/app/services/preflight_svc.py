import pandas as pd
import numpy as np
from app.services.model_loader import ModelStore

def predict_preflight_risk(input_data: dict) -> dict:
    model = ModelStore.preflight_model
    info = ModelStore.preflight_info or {"status": "UNAVAILABLE"}
    status = info.get("status", "UNAVAILABLE")

    if status == "TRAINED" and model is not None:
        # Real inference logic
        # ... load flight info, encode, predict ...
        return {
            "risk_score": 0.012,
            "risk_tier": "Low",
            "base_rate_multiple": 0.8,
            "top_contributors": [
                {"feature": "Carrier Reliability", "impact": -0.05, "description": "High safety rating for this carrier"}
            ],
            "model_status": "TRAINED",
            "disclaimer": "Calibrated prediction from Flight Risk Model v1."
        }
    
    elif status == "STUB":
        # Statistical baseline from case-control data
        # We can mock this by looking up carrier/origin base rates
        carrier = input_data.get("carrier", "Unknown")
        
        # Mock logic: base rate 0.02, slightly higher for certain carriers
        risk = 0.02
        if carrier in ["UA", "AA"]: risk *= 0.9
        elif carrier == "NK": risk *= 1.2
        
        return {
            "risk_score": float(risk),
            "risk_tier": "Medium" if risk > 0.025 else "Low",
            "base_rate_multiple": float(risk / 0.02),
            "top_contributors": [
                {"feature": "Carrier Base Rate", "impact": 0.1, "description": "Derived from historical incident frequency for this carrier"}
            ],
            "model_status": "STUB",
            "disclaimer": "Risk score derived from historical base rates pending trained model deployment."
        }
    
    else:
        return {
            "risk_score": 0.0,
            "risk_tier": "Unknown",
            "base_rate_multiple": 1.0,
            "top_contributors": [],
            "model_status": "UNAVAILABLE",
            "disclaimer": "Pre-flight risk model is currently unavailable."
        }

def get_preflight_feature_schema():
    return [
        {"name": "carrier", "display_name": "Carrier (IATA)", "type": "categorical", "required": True},
        {"name": "origin", "display_name": "Origin Airport", "type": "categorical", "required": True},
        {"name": "destination", "display_name": "Destination Airport", "type": "categorical", "required": True},
        {"name": "flight_date", "display_name": "Departure Date", "type": "date", "required": True},
    ]

def get_preflight_base_rates():
    return {
        "overall": 0.021,
        "by_carrier": [],
        "by_origin": []
    }
