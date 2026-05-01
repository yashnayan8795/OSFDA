import sys
import os
sys.path.insert(0, r"e:\OSFDA")

try:
    from streamlit_app.utils.loaders import load_severity_test_predictions
    y_true, y_pred, cal_probs, X_test, feature_cols = load_severity_test_predictions()
    from sklearn.metrics import cohen_kappa_score, classification_report
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    cls = classification_report(y_true, y_pred, output_dict=True)
    print(f"Problem A -> QWK: {qwk:.4f}, Macro-F1: {cls['macro avg']['f1-score']:.4f}")
except Exception as e:
    print(f"Problem A -> Error: {e}")

try:
    from src.evaluation.multilabel_metrics import evaluate_multilabel
    # The models might already have evaluation metrics stored in joblib.
    import joblib
    sbert_model = joblib.load(r"e:\OSFDA\models\category_text_tower.joblib")
    if "metrics" in sbert_model:
        print(f"Problem B (SBERT) metrics: {sbert_model['metrics']}")
    
    tfidf_model = joblib.load(r"e:\OSFDA\models\category_tfidf_baseline.joblib")
    if "metrics" in tfidf_model:
        print(f"Problem B (TF-IDF) metrics: {tfidf_model['metrics']}")
except Exception as e:
    print(f"Problem B -> Error: {e}")

try:
    import pandas as pd
    df = pd.read_parquet(r"e:\OSFDA\data\processed\preflight_features_final.parquet")
    if "incident" in df.columns:
        pos_rate = df["incident"].mean()
        print(f"Problem C -> Positive Rate in Data: {pos_rate:.6f} ({pos_rate*100:.4f}%)")
except Exception as e:
    print(f"Problem C -> Error: {e}")
