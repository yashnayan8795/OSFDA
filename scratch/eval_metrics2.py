import sys
sys.path.insert(0, r"e:\OSFDA")

try:
    import joblib
    sbert_model = joblib.load(r"e:\OSFDA\models\category_text_tower.joblib")
    if "metrics" in sbert_model:
        print(f"Problem B (SBERT) metrics: {sbert_model['metrics']}")
    
    tfidf_model = joblib.load(r"e:\OSFDA\models\category_tfidf_baseline.joblib")
    if "metrics" in tfidf_model:
        print(f"Problem B (TF-IDF) metrics: {tfidf_model['metrics']}")
except Exception as e:
    print(f"Problem B -> Error: {e}")

