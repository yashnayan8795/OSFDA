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

def label_bts_incidents(bts: pd.DataFrame, ntsb_keys: Set[Tuple], window_days: int = 3) -> pd.DataFrame:
    """
    Flag BTS flights that occurred near an NTSB incident day (within window_days) at the same airport.
    """
    bts = bts.copy()
    bts['match_date'] = pd.to_datetime(bts['FL_DATE']).dt.date
    
    # Map airport -> list of incident dates for O(1) lookup
    airport_incidents = {}
    for date, airport in ntsb_keys:
        if pd.isna(date) or pd.isna(airport):
            continue
        airport = str(airport).strip().upper()
        if airport not in airport_incidents:
            airport_incidents[airport] = []
        airport_incidents[airport].append(date)
    
    # Convert dates to list of pd.Timestamp objects for distance calculation
    for apt in airport_incidents:
        airport_incidents[apt] = pd.to_datetime(airport_incidents[apt])
        
    is_incident_day = []
    for d, a in zip(bts['match_date'], bts['ORIGIN']):
        if pd.isna(d) or pd.isna(a):
            is_incident_day.append(False)
            continue
        a = str(a).strip().upper()
        if a not in airport_incidents:
            is_incident_day.append(False)
            continue
        
        # Check if any incident date is within window_days
        flight_date = pd.to_datetime(d)
        dates = airport_incidents[a]
        diffs = np.abs((dates - flight_date).days)
        is_incident_day.append(np.any(diffs <= window_days))
        
    bts['is_incident_day'] = is_incident_day
    return bts

