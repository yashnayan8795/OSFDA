import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path
import joblib

from src.features.preflight import engineer_preflight_features

def main():
    print("Loading data...")
    df = pd.read_parquet("data/processed/preflight_features_final.parquet")
    
    print("Engineering features...")
    df = engineer_preflight_features(df)
    
    # Features for CatBoost (using the original categoricals)
    cat_features = ['ORIGIN', 'DEST', 'OP_UNIQUE_CARRIER']
    
    # Fill NAs in categoricals
    for c in cat_features:
        if c in df.columns:
            df[c] = df[c].fillna('UNKNOWN').astype(str)
            
    num_features = [
        'month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'hour', 
        'airport_risk_rate', 'carrier_risk_rate', 'route_risk_rate', 'carrier_route_risk_rate',
        'DISTANCE', 'adverse_weather_index'
    ]
    
    features = cat_features + num_features
    
    if 'year' not in df.columns:
        for date_col in ['FL_DATE', 'fl_date', 'date', 'Date']:
            if date_col in df.columns:
                df['year'] = pd.to_datetime(df[date_col]).dt.year
                break
                
    if 'incident_confidence' not in df.columns:
        df['incident_confidence'] = df['incident']
        
    train_df = df[df['year'] == 2018]
    val_df   = df[df['year'] == 2019]
    test_df  = df[df['year'] == 2020]
    
    X_train, y_train = train_df[features], train_df['incident']
    w_train = np.where(y_train == 1, train_df['incident_confidence'], 1.0)
    
    X_val, y_val     = val_df[features], val_df['incident']
    w_val = np.where(y_val == 1, val_df['incident_confidence'], 1.0)
    
    X_test, y_test   = test_df[features], test_df['incident']
    
    print("Training CatBoost model...")
    cb_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_features,
        auto_class_weights='Balanced',
        eval_metric='AUC',
        random_seed=42,
        verbose=100
    )
    
    cb_model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50
    )
    
    print("Evaluating Test Set with CatBoost...")
    cb_probs = cb_model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, cb_probs)
    pr_auc = average_precision_score(y_test, cb_probs)
    
    print(f"CatBoost Test ROC-AUC: {roc_auc:.4f}")
    print(f"CatBoost Test PR-AUC:  {pr_auc:.4f}")
    
    n_decile = max(int(len(y_test) * 0.10), 1)
    top_indices = np.argsort(cb_probs)[::-1][:n_decile]
    top_decile_precision = y_test.iloc[top_indices].mean()
    overall_precision = y_test.mean()
    lift_top_decile = top_decile_precision / max(overall_precision, 1e-5)
    print(f"CatBoost Lift@10%: {lift_top_decile:.2f}x")
    
    # Try Ensemble with LightGBM
    lgb_path = Path("models/preflight_lgbm_phase7.joblib")
    if lgb_path.exists():
        print("Found LightGBM model. Evaluating Ensemble...")
        lgb_artifact = joblib.load(lgb_path)
        lgb_model = lgb_artifact['model']
        lgb_features = lgb_artifact['features']
        
        # Prepare LGBM test set
        X_test_lgb = test_df[lgb_features]
        lgb_probs = lgb_model.predict_proba(X_test_lgb)[:, 1]
        
        ens_probs = 0.6 * cb_probs + 0.4 * lgb_probs
        ens_roc_auc = roc_auc_score(y_test, ens_probs)
        ens_pr_auc = average_precision_score(y_test, ens_probs)
        
        top_indices_ens = np.argsort(ens_probs)[::-1][:n_decile]
        ens_lift = y_test.iloc[top_indices_ens].mean() / max(overall_precision, 1e-5)
        
        print(f"Ensemble Test ROC-AUC: {ens_roc_auc:.4f}")
        print(f"Ensemble Test PR-AUC:  {ens_pr_auc:.4f}")
        print(f"Ensemble Lift@10%: {ens_lift:.2f}x")

if __name__ == '__main__':
    main()
