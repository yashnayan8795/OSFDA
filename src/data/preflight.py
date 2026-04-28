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

def create_match_keys(ntsb: pd.DataFrame) -> Set[Tuple]:
    """
        Extract unique (date, airport) pairs from NTSB incidents.
    """
    ntsb = ntsb.copy()
    ntsb['event_date'] = pd.to_datetime(ntsb['event_date']).dt.date
    # FAA LID to IATA normalization (simple upper for POC)
    ntsb['airport_code'] = ntsb['arpt_id'].str.strip().str.upper()
    
    keys = set(zip(ntsb['event_date'], ntsb['airport_code']))
    return {k for k in keys if pd.notna(k[0]) and pd.notna(k[1])}

def label_bts_incidents(bts: pd.DataFrame, ntsb_keys: Set[Tuple]) -> pd.DataFrame:
    """
        Flag BTS flights that occurred on an NTSB incident day at the same airport.
    """
    bts = bts.copy()
    bts['match_date'] = pd.to_datetime(bts['FL_DATE']).dt.date
    
    bts['is_incident_day'] = [
        (d, a) in ntsb_keys 
        for d, a in zip(bts['match_date'], bts['ORIGIN'])
    ]
    return bts
