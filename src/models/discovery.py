"""
Problem D — Emerging Risk Discovery (Unsupervised)
===================================================
1. Topic Modeling (BERTopic on SBERT embeddings)
2. Temporal Dynamics (Changepoint Detection via PELT)
3. Emerging Risk Scoring
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import joblib
from pathlib import Path

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
    from sklearn.cluster import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    # PCA for dimensionality reduction (bypassing UMAP/Numba issues)
    umap_model = PCA(n_components=15, random_state=42)

    # Use sklearn's built-in HDBSCAN to restore density-based clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric='euclidean',
        cluster_selection_method='eom'
    )

    # CountVectorizer for topic representation
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        language="english",
        calculate_probabilities=False,
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
    time_col: str = "Time_Date"
) -> pd.DataFrame:
    """
    Calculates monthly prevalence for each topic.
    """
    df = df.copy()
    df['topic'] = topics
    
    # Exclude outliers (-1)
    df = df[df['topic'] != -1]
    
    # Create period
    if "year" in df.columns and "month" in df.columns:
        df['period'] = df['year'].astype(str) + "-" + df['month'].astype(str).str.zfill(2)
    else:
        df['period'] = df[time_col].astype(str)
        
    monthly_counts = df.groupby(['topic', 'period']).size().unstack(fill_value=0)
    return monthly_counts

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
            result = algo.predict(pen=penalty)
            # ruptures returns breakpoints including the end of the signal
            changepoints[topic_idx] = [cp for cp in result if cp < len(signal)]
        except Exception:
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
    
    for topic_idx in monthly_counts.index:
        signal = monthly_counts.loc[topic_idx].values
        cps = changepoints.get(topic_idx, [])
        
        # Recent growth: compare last 6 months to previous 12 months (if enough data)
        if len(signal) >= 18:
            recent_avg = np.mean(signal[-6:])
            past_avg = np.mean(signal[-18:-6])
            growth = recent_avg / (past_avg + 0.1)
        else:
            growth = 1.0
            
        # Is there a recent changepoint? (in the last 12 periods)
        recent_cp = any((len(signal) - cp) <= 12 for cp in cps)
        
        severity = severity_by_topic.get(topic_idx, 0.0)
        
        # Score calculation
        score = growth * (1 + 0.5 * severity) * (1.5 if recent_cp else 1.0)
        
        results.append({
            'Topic': topic_idx,
            'Growth_Ratio': growth,
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
