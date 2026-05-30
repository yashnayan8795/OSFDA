import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from src.data.preflight import label_bts_incidents

def main():
    print("Loading data...")
    # Load raw data
    ntsb = pd.read_parquet("data/raw/ntsb_accidents.parquet")
    
    # Load processed features to get the flights (it already has weather joined etc)
    df = pd.read_parquet("data/processed/preflight_features_final.parquet")
    
    # Remove old incident columns if they exist
    if 'incident' in df.columns:
        df = df.drop(columns=['incident'])
    if 'incident_confidence' in df.columns:
        df = df.drop(columns=['incident_confidence'])
    if 'is_incident_day' in df.columns:
        df = df.drop(columns=['is_incident_day'])
        
    print("Relabeling BTS flights with hierarchical confidence...")
    df = label_bts_incidents(df, ntsb)
    
    # Map back to 'incident'
    df['incident'] = df['is_incident_day'].astype(int)
    
    print(f"New incident rate: {df['incident'].mean():.4f}")
    print("Confidence score distribution:")
    print(df['incident_confidence'].value_counts())
    
    # Save back to processed
    out_path = "data/processed/preflight_features_final.parquet"
    df.to_parquet(out_path)
    print(f"Saved updated labels to {out_path}")

if __name__ == '__main__':
    main()
