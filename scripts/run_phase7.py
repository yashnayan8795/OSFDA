import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import joblib
from pathlib import Path

from src.models.preflight import PriorShiftedCalibratedModel
from src.features.preflight import engineer_preflight_features

def main():
    print("Loading data...")
    df = pd.read_parquet("data/processed/preflight_features_final.parquet")
    
    # Run our new advanced feature engineering
    print("Engineering features...")
    df = engineer_preflight_features(df)
    
    features = [
        'month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'hour', 
        'temp', 'rhum', 'prcp', 'wspd', 'pres',
        'airport_risk_rate', 'carrier_risk_rate', 'route_risk_rate', 'carrier_route_risk_rate',
        'DISTANCE',
        'adverse_weather_index'
    ]
    
    # Handle missing year
    if 'year' not in df.columns:
        for date_col in ['FL_DATE', 'fl_date', 'date', 'Date']:
            if date_col in df.columns:
                df['year'] = pd.to_datetime(df[date_col]).dt.year
                break
                
    # Define sample weights logic: 
    # use incident_confidence for positive class, 1.0 for negative class
    # but also maintain class balancing weight because of severe imbalance!
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
    
    train_prior = y_train.mean()
    if train_prior == 0:
        train_prior = 0.0476
    true_prior = 0.0476  # Using actual historical positive rate
    
    print(f"Train prior: {train_prior:.4f}, True target prior: {true_prior:.4f}")
    
    print("Training LightGBM model...")
    # scale_pos_weight combined with our sample weight
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=20.0,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    print("Calibrating model via Isotonic Regression...")
    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_val, y_val, sample_weight=w_val)
    
    print("Applying Prior Probability Shift...")
    shifted_model = PriorShiftedCalibratedModel(calibrated_model, true_prior, train_prior, features)
    
    print("Evaluating Test Set...")
    probs = shifted_model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print(f"Test PR-AUC:  {pr_auc:.4f}")
    print(f"Average Predicted Prob: {probs.mean():.6f}")
    
    # Ranking metrics
    n_decile = max(int(len(y_test) * 0.10), 1)
    top_indices = np.argsort(probs)[::-1][:n_decile]
    top_decile_precision = y_test.iloc[top_indices].mean()
    overall_precision = y_test.mean()
    lift_top_decile = top_decile_precision / max(overall_precision, 1e-5)
    print(f"Lift@10%: {lift_top_decile:.2f}x")
    
    artifact = {
        "model": shifted_model,
        "features": features,
        "true_prior": true_prior
    }
    
    out_path = Path("models/preflight_lgbm_phase7.joblib")
    joblib.dump(artifact, out_path)
    print(f"Model saved to {out_path}")

if __name__ == '__main__':
    main()
