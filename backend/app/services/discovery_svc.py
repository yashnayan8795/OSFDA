import pandas as pd
from app.services.model_loader import ModelStore

def get_emerging_risks(limit: int = 10, sort_by: str = "risk_score") -> list[dict]:
    df = ModelStore.emerging_risks_df
    if df is None:
        return []
    
    # Map sort enum to CSV columns
    sort_map = {
        "risk_score": "Risk_Score",
        "growth": "Growth_Ratio",
        "severity": "Avg_Severity",
        "count": "Count"
    }
    col = sort_map.get(sort_by, "Risk_Score")
    
    top = df.sort_values(col, ascending=False).head(limit)
    
    results = []
    for _, row in top.iterrows():
        results.append({
            "topic_id": int(row["Topic"]),
            "name": row["Name"],
            "risk_score": float(row["Risk_Score"]),
            "growth_ratio": float(row["Growth_Ratio"]),
            "recent_changepoint": bool(row["Recent_Changepoint"]),
            "avg_severity": float(row["Avg_Severity"]),
            "count": int(row["Count"])
        })
    return results

def get_risk_trend(topic_id: int) -> list[dict]:
    df = ModelStore.topic_trends_df
    if df is None:
        return []
    
    subset = df[df["topic_id"] == topic_id].sort_values("period")
    return [
        {"period": row["period"], "count": int(row["count"]), "avg_severity": float(row["avg_severity"])}
        for _, row in subset.iterrows()
    ]

def get_topic_detail(topic_id: int) -> dict:
    df = ModelStore.emerging_risks_df
    if df is None: return {}
    
    row = df[df["Topic"] == topic_id]
    if row.empty: return {}
    row = row.iloc[0]
    
    reps = ModelStore.topic_representations or {}
    topic_rep = reps.get(str(topic_id), reps.get(topic_id, {}))
    
    return {
        "topic_id": topic_id,
        "name": row["Name"],
        "keywords": topic_rep.get("keywords", []),
        "sample_reports": topic_rep.get("sample_reports", []),
        "description": f"Safety topic identified via BERTopic. Top keywords: {', '.join(topic_rep.get('keywords', [])[:5])}"
    }

def get_changepoint_alerts() -> list[dict]:
    df = ModelStore.emerging_risks_df
    if df is None: return []
    
    # Filter only topics with recent changepoints
    alerts_df = df[df["Recent_Changepoint"] == True]
    
    results = []
    for _, row in alerts_df.iterrows():
        results.append({
            "topic_id": int(row["Topic"]),
            "name": row["Name"],
            "date": "2023-06", # POC date
            "severity_association": float(row["Avg_Severity"]),
            "direction": "UP" if row["Growth_Ratio"] > 1 else "DOWN"
        })
    return results
