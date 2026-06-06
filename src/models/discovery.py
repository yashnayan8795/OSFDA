"""
Problem D — Emerging Risk Discovery (Unsupervised)
===================================================
1. Topic Modeling (BERTopic on SBERT embeddings)
2. Temporal Dynamics (Changepoint Detection via PELT)
3. Emerging Risk Scoring
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

def fit_bertopic(
    texts: pd.Series,
    embeddings: np.ndarray,
    min_topic_size: int = 150,
) -> Any:
    """
    Fits a BERTopic model on the provided texts and embeddings.
    """
    from bertopic import BERTopic
    from sklearn.decomposition import PCA
    import hdbscan
    from sklearn.feature_extraction.text import CountVectorizer

    # PCA for dimensionality reduction (bypassing UMAP/Numba DLL policy block)
    umap_model = PCA(n_components=15, random_state=42)

    # Use HDBSCAN for better cluster geometries and outlier detection
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=50, metric='euclidean', cluster_selection_method='eom', prediction_data=True
    )

    # CountVectorizer for topic representation
    vectorizer_model = CountVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=5
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        language="english",
        calculate_probabilities=False,
        nr_topics=30,
        verbose=True
    )

    print("  [BERTopic] Fitting model...")
    topics, probs = topic_model.fit_transform(texts.tolist(), embeddings)
    return topic_model, topics

def get_topic_info(topic_model) -> pd.DataFrame:
    return topic_model.get_topic_info()

def calculate_temporal_trends(
    df: pd.DataFrame,
    topics: List[int],
    time_col: str = "Time_Date",
) -> pd.DataFrame:
    """
    Calculates monthly prevalence and average severity for each topic.

    Returns two objects:
      - monthly_counts : wide DataFrame (topics x periods) of report counts —
        used internally by detect_changepoints and score_emerging_risks.
      - trends_long    : long DataFrame with columns
        [topic_id, period, count, avg_severity] — saved to topic_trends.parquet.

    For backwards-compat the function still returns monthly_counts; call
    calculate_temporal_trends_long() to get the Streamlit-ready long format.
    """
    df = df.copy()
    df["topic"] = topics
    df = df[df["topic"] != -1]

    if "year" in df.columns and "month" in df.columns:
        df["period"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    else:
        df["period"] = df[time_col].astype(str)

    monthly_counts = df.groupby(["topic", "period"]).size().unstack(fill_value=0)
    return monthly_counts


def calculate_temporal_trends_long(
    df: pd.DataFrame,
    topics: List[int],
    time_col: str = "Time_Date",
) -> pd.DataFrame:
    """
    Long-format version of calculate_temporal_trends.
    Returns DataFrame with [topic_id, period, count, avg_severity] — one row
    per (topic, period) combination. avg_severity is computed per period, not
    globally, so the Streamlit trend charts show severity evolution over time.
    """
    df = df.copy()
    df["topic"] = topics
    df = df[df["topic"] != -1]

    if "year" in df.columns and "month" in df.columns:
        df["period"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    else:
        df["period"] = df[time_col].astype(str)

    agg: Dict[str, Any] = {"count": ("topic", "size")}
    if "severity_level" in df.columns:
        agg["avg_severity"] = ("severity_level", "mean")

    grouped = df.groupby(["topic", "period"]).agg(**agg).reset_index()
    grouped = grouped.rename(columns={"topic": "topic_id"})

    if "avg_severity" not in grouped.columns:
        grouped["avg_severity"] = 0.0

    return grouped

def detect_changepoints(
    monthly_counts: pd.DataFrame,
    penalty: float = 3.0
) -> Dict[int, List[int]]:
    """
    Detects structural shifts in monthly topic frequency using PELT.
    """
    import ruptures as rpt
    
    changepoints = {}
    print(f"  [Changepoints] Analyzing {len(monthly_counts)} topics...")
    for topic_idx in monthly_counts.index:
        signal = monthly_counts.loc[topic_idx].values
        if len(signal) < 10:
            changepoints[topic_idx] = []
            continue
            
        # Ensure 2D for ruptures
        signal_2d = signal.reshape(-1, 1)
        
        try:
            algo = rpt.Pelt(model="rbf").fit(signal_2d)
            
            # Dynamic penalty based on signal variance and length (BIC-style heuristic)
            signal_var = np.var(signal)
            dynamic_pen = np.log(len(signal)) * max(1.0, signal_var) if signal_var > 0 else penalty
            # Bound the penalty to reasonable limits to prevent over-segmentation or zero breaks
            dynamic_pen = max(1.0, min(10.0, dynamic_pen))
            
            result = algo.predict(pen=dynamic_pen)
            # ruptures returns breakpoints including the end of the signal
            changepoints[topic_idx] = [cp for cp in result if cp < len(signal)]
        except Exception as exc:
            logger.warning("PELT failed for topic %s: %s", topic_idx, exc)
            changepoints[topic_idx] = []
        
    return changepoints

def score_emerging_risks(
    topic_info: pd.DataFrame,
    monthly_counts: pd.DataFrame,
    changepoints: Dict[int, List[int]],
    df: pd.DataFrame,
    topics: List[int]
) -> pd.DataFrame:
    """
    Computes an Emerging Risk Score combining growth, changepoints, and severity.
    """
    df_temp = df.copy()
    df_temp['topic'] = topics
    
    # Calculate average severity per topic
    if 'severity_level' in df_temp.columns:
        severity_by_topic = df_temp[df_temp['topic'] != -1].groupby('topic')['severity_level'].mean()
    else:
        severity_by_topic = pd.Series(0.0, index=topic_info['Topic'])

    results = []
    
    from scipy.stats import poisson
    
    for topic_idx in monthly_counts.index:
        signal = monthly_counts.loc[topic_idx].values
        cps = changepoints.get(topic_idx, [])
        
        # Recent growth: compare last 6 months to previous 12 months (if enough data)
        if len(signal) >= 18:
            recent_counts = signal[-6:]
            past_counts = signal[-18:-6]
            recent_avg = np.mean(recent_counts)
            past_avg = np.mean(past_counts)
            growth = recent_avg / (past_avg + 0.1)
            
            # Poisson rate test p-value:
            # Null hypothesis: recent counts are generated by the past_avg Poisson rate
            recent_sum = np.sum(recent_counts)
            expected_sum = past_avg * len(recent_counts)
            # p-value of seeing >= recent_sum given expected_sum
            p_val = poisson.sf(recent_sum - 1, expected_sum)
        else:
            growth = 1.0
            p_val = 1.0
            
        # Is there a recent changepoint? (in the last 12 periods)
        recent_cp = any((len(signal) - cp) <= 12 for cp in cps)
        
        severity = severity_by_topic.get(topic_idx, 0.0)
        
        # Score calculation with statistical grounding
        # We boost the score if growth is statistically significant (p < 0.05), otherwise dampen it
        significance_multiplier = 1.5 if p_val < 0.05 else 0.5
        score = growth * (1 + 0.5 * severity) * (1.5 if recent_cp else 1.0) * significance_multiplier
        
        results.append({
            'Topic': topic_idx,
            'Growth_Ratio': growth,
            'P_Value': p_val,
            'Recent_Changepoint': recent_cp,
            'Avg_Severity': severity,
            'Risk_Score': score
        })
        
    scores_df = pd.DataFrame(results)
    
    # Merge with topic names and filter out outliers (-1) if any are left
    final_df = pd.merge(scores_df, topic_info[['Topic', 'Name', 'Representation', 'Count']], on='Topic')
    final_df = final_df[final_df['Topic'] != -1].sort_values('Risk_Score', ascending=False)
    
    return final_df

def save_discovery_model(model: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    print(f"  Saved BERTopic model to {path}")
    return path
