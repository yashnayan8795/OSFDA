import pandas as pd
import numpy as np
from app.services.model_loader import ModelStore
from src.models.category import predict_tfidf

LABEL_DISPLAY = {
    "Flight_Operations": "Flight Operations",
    "Equipment_System": "Equipment / System",
    "ATC_Communication": "ATC / Communication",
    "Environment": "Environment",
    "Airspace_Navigation": "Airspace / Navigation",
}

def classify_categories(narrative: str, synopsis: str = "", accurate: bool = False) -> dict:
    combined = f"{narrative} {synopsis}".strip()
    text_series = pd.Series([combined])
    
    # Check if fusion is requested and available
    if accurate and ModelStore.category_fusion is not None:
        # Fusion inference (requires SBERT encoding first)
        # For simplicity in POC, we'll use TF-IDF if SBERT environment isn't fully ready
        # But logically this is where it would live:
        # from src.models.category import encode_texts
        # emb = encode_texts(text_series)
        # probs, preds = ModelStore.category_fusion.predict(emb, ...)
        model = ModelStore.category_tfidf # Fallback for now
    else:
        model = ModelStore.category_tfidf

    if model is None:
        return {"predictions": []}

    probs, preds = predict_tfidf(model, text_series)
    label_names = model["label_names"]
    thresholds = model["thresholds"]

    predictions = []
    for i, label in enumerate(label_names):
        predictions.append({
            "label": label,
            "display_name": LABEL_DISPLAY.get(label, label),
            "probability": float(probs[0, i]),
            "predicted": bool(preds[0, i]),
            "threshold_used": float(thresholds.get(label, 0.5))
        })

    # Sort by probability descending
    predictions.sort(key=lambda x: x["probability"], reverse=True)
    return {"predictions": predictions}

def get_category_labels():
    model = ModelStore.category_tfidf
    if model is None: return []
    
    label_names = model["label_names"]
    thresholds = model["thresholds"]
    
    return [
        {
            "label": l,
            "display_name": LABEL_DISPLAY.get(l, l),
            "threshold": float(thresholds.get(l, 0.5))
        }
        for l in label_names
    ]

def get_category_distribution():
    # Mock distribution from training corpus
    return [
        {"label": "Flight_Operations", "count": 22000},
        {"label": "Equipment_System", "count": 12000},
        {"label": "ATC_Communication", "count": 8500},
        {"label": "Environment", "count": 6000},
        {"label": "Airspace_Navigation", "count": 5200},
    ]
