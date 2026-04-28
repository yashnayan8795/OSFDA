import sys
from pathlib import Path
import pandas as pd
import json
import joblib

# Ensure src.* is importable if not already handled by main.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

class ModelStore:
    ready: bool = False

    # Section A - Severity
    severity_model = None
    severity_calibrators = None
    severity_feature_cols = None
    severity_info = None

    # Section B - Category
    category_tfidf = None
    category_fusion = None
    category_info = None

    # Section C - Preflight
    preflight_model = None
    preflight_info = None

    # Section D - Discovery
    emerging_risks_df = None
    topic_trends_df = None
    topic_representations = None
    topic_changepoints = None

    # Section E - Graph
    graph_nodes = None
    graph_edges = None
    centrality_df = None
    factor_patterns = None

    @classmethod
    def load(cls):
        print("[ModelStore] Loading artifacts...")
        cls._load_severity()
        cls._load_category()
        cls._load_preflight()
        cls._load_discovery()
        cls._load_graph()
        cls.ready = True
        print("[ModelStore] All artifacts loaded.")

    @classmethod
    def _load_severity(cls):
        from catboost import CatBoostClassifier
        model_path = MODELS_DIR / "severity_catboost.cbm"
        if model_path.exists():
            model = CatBoostClassifier()
            model.load_model(str(model_path))
            cls.severity_model = model
            print(f"  [A] Severity model loaded")
        
        calib_path = MODELS_DIR / "severity_calibrators.joblib"
        if calib_path.exists():
            cls.severity_calibrators = joblib.load(calib_path)
            
        feat_path = MODELS_DIR / "severity_feature_cols.json"
        if feat_path.exists():
            with open(feat_path) as f:
                cls.severity_feature_cols = json.load(f)
                
        info_path = MODELS_DIR / "severity_model_info.json"
        if info_path.exists():
            with open(info_path) as f:
                cls.severity_info = json.load(f)

    @classmethod
    def _load_category(cls):
        tfidf_path = MODELS_DIR / "category_tfidf_baseline.joblib"
        if tfidf_path.exists():
            cls.category_tfidf = joblib.load(tfidf_path)
            print(f"  [B] Category TF-IDF loaded")
            
        fusion_path = MODELS_DIR / "category_fusion.joblib"
        if fusion_path.exists():
            cls.category_fusion = joblib.load(fusion_path)
            print(f"  [B] Category Fusion loaded")
            
        info_path = MODELS_DIR / "category_model_info.json"
        if info_path.exists():
            with open(info_path) as f:
                cls.category_info = json.load(f)

    @classmethod
    def _load_preflight(cls):
        # Check for trained model
        model_path = MODELS_DIR / "preflight_lgbm.joblib"
        if model_path.exists():
            cls.preflight_model = joblib.load(model_path)
            
        info_path = MODELS_DIR / "preflight_model_info.json"
        if info_path.exists():
            with open(info_path) as f:
                cls.preflight_info = json.load(f)

    @classmethod
    def _load_discovery(cls):
        risks_path = DATA_DIR / "emerging_risks.csv"
        if risks_path.exists():
            cls.emerging_risks_df = pd.read_csv(risks_path)
            
        trends_path = DATA_DIR / "topic_trends.parquet"
        if trends_path.exists():
            cls.topic_trends_df = pd.read_parquet(trends_path)
            
        reps_path = DATA_DIR / "topic_representations.json"
        if reps_path.exists():
            with open(reps_path) as f:
                cls.topic_representations = json.load(f)
                
        cps_path = DATA_DIR / "topic_changepoints.json"
        if cps_path.exists():
            with open(cps_path) as f:
                cls.topic_changepoints = json.load(f)
        print(f"  [D] Discovery artifacts loaded")

    @classmethod
    def _load_graph(cls):
        graph_path = DATA_DIR / "factor_graph.json"
        if graph_path.exists():
            with open(graph_path) as f:
                raw = json.load(f)
            cls.graph_nodes = raw.get("nodes", [])
            cls.graph_edges = raw.get("links", [])
            
        centrality_path = DATA_DIR / "centrality_report.csv"
        if centrality_path.exists():
            cls.centrality_df = pd.read_csv(centrality_path)
            
        patterns_path = DATA_DIR / "factor_patterns.json"
        if patterns_path.exists():
            with open(patterns_path) as f:
                cls.factor_patterns = json.load(f)
        print(f"  [E] Graph artifacts loaded")
