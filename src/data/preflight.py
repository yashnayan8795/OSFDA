"""
    Problem C — Data Utilities
    ==========================
    Data loading and case-control matching logic for pre-flight risk prediction.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Set

def load_preflight_raw(data_dir: Path = Path('data/raw')) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
        Load NTSB and BTS raw parquets.
    """
    ntsb = pd.read_parquet(data_dir / 'ntsb_accidents.parquet')
    bts_files = sorted((data_dir / 'bts_flights').glob('bts_20??.parquet'))
    bts = pd.concat([pd.read_parquet(f) for f in bts_files], ignore_index=True)
    return ntsb, bts

import numpy as np

def label_bts_incidents(bts: pd.DataFrame, ntsb: pd.DataFrame) -> pd.DataFrame:
    """
    Flag BTS flights that occurred near an NTSB incident, assigning a confidence score
    based on the strength of the match (Origin, Dest, Time Window).
    """
    bts = bts.copy()
    bts['match_date'] = pd.to_datetime(bts['FL_DATE']).dt.date
    
    # Preprocess NTSB data for matching
    ntsb_clean = ntsb.copy()
    ntsb_clean['event_date'] = pd.to_datetime(ntsb_clean['event_date']).dt.date
    
    # Extract available airports
    ntsb_clean['ev_apt'] = ntsb_clean['ev_nr_apt_id'].str.strip().str.upper()
    ntsb_clean['dprt_apt'] = ntsb_clean['dprt_apt_id'].str.strip().str.upper()
    ntsb_clean['dest_apt'] = ntsb_clean['dest_apt_id'].str.strip().str.upper()
    
    # Convert NTSB to list of dicts for faster iteration
    ntsb_records = ntsb_clean[['event_date', 'ev_apt', 'dprt_apt', 'dest_apt']].to_dict('records')
    
    confidence_scores = []
    is_incident = []
    
    for _, row in bts.iterrows():
        b_date = row['match_date']
        b_orig = str(row['ORIGIN']).strip().upper()
        b_dest = str(row.get('DEST', '')).strip().upper()
        
        if pd.isna(b_date) or pd.isna(b_orig):
            confidence_scores.append(0.0)
            is_incident.append(False)
            continue
            
        best_conf = 0.0
        
        for n in ntsb_records:
            n_date = n['event_date']
            if pd.isna(n_date):
                continue
                
            day_diff = abs((n_date - b_date).days)
            if day_diff > 3:
                continue
                
            # Check Origin match (either explicit departure or event location)
            origin_match = (b_orig == n['dprt_apt']) or (b_orig == n['ev_apt'])
            if not origin_match:
                continue
                
            dest_match = (b_dest == n['dest_apt']) and pd.notna(n['dest_apt']) and n['dest_apt'] != ''
            
            # Apply hierarchical rules
            # Very High (1.0): ORIGIN + DEST + ±1 day
            if origin_match and dest_match and day_diff <= 1:
                conf = 1.0
            # High (0.8): ORIGIN + DEST + ±3 day
            elif origin_match and dest_match and day_diff <= 3:
                conf = 0.8
            # Medium (0.6): ORIGIN only + ±1 day
            elif origin_match and day_diff <= 1:
                conf = 0.6
            # Low (0.3): ORIGIN only + ±3 day
            elif origin_match and day_diff <= 3:
                conf = 0.3
            else:
                conf = 0.0
                
            if conf > best_conf:
                best_conf = conf
                if best_conf == 1.0:
                    break # can't get better
                    
        confidence_scores.append(best_conf)
        is_incident.append(best_conf > 0.0)
        
    bts['incident_confidence'] = confidence_scores
    bts['is_incident_day'] = is_incident
    return bts


