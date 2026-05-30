import pandas as pd
import numpy as np
from typing import List, Optional

def compute_historical_rates(
    df: pd.DataFrame,
    group_col: str,
    target_col: str = 'incident',
    decay_halflife_days: Optional[int] = None,
    date_col: str = 'FL_DATE'
) -> pd.Series:
    """
    Compute rolling historical incident rates for a given entity.
    Avoids leakage using shift(1).
    If decay_halflife_days is provided, uses an exponentially weighted moving average.
    """
    df = df.sort_values([group_col, date_col])

    if decay_halflife_days is not None:
        rates = df.groupby(group_col)[target_col].transform(
            lambda x: x.ewm(halflife=f'{decay_halflife_days} days', 
                            times=pd.to_datetime(df.loc[x.index, date_col])).mean().shift(1)
        )
    else:
        rates = df.groupby(group_col)[target_col].transform(
            lambda x: x.expanding().mean().shift(1)
        )

    return rates


def add_temporal_features(df: pd.DataFrame, date_col: str = 'FL_DATE') -> pd.DataFrame:
    """
    Extract cyclical temporal features.
    """
    df = df.copy()
    dates = pd.to_datetime(df[date_col])

    df['month'] = dates.dt.month
    df['day_of_week'] = dates.dt.dayofweek

    df['DEP_TIME'] = df['DEP_TIME'].fillna(0).astype(int)
    df['hour'] = (df['DEP_TIME'] // 100).astype(int)

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    return df


def categorize_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create categorical buckets for weather metrics.
    """
    df = df.copy()

    if 'wspd' in df.columns:
        df['wind_cat'] = pd.cut(
            df['wspd'],
            bins=[-1, 5, 15, 30, 50, 200],
            labels=['Calm', 'Light', 'Moderate', 'Strong', 'Gale']
        )
    else:
        df['wind_cat'] = 'Unknown'

    return df


def engineer_preflight_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer advanced pre-flight risk features including:
    - Route, Carrier, Airport, and Carrier-Route expanding risk rates (with time decay)
    - NOAA weather severity indicators and Adverse Weather Index
    """
    df = df.copy()
    
    # 1. Historical Risk Rates
    if 'route_id' not in df.columns and 'ORIGIN' in df.columns and 'DEST' in df.columns:
        df['route_id'] = df['ORIGIN'] + '_' + df['DEST']
        
    if 'carrier_route_id' not in df.columns and 'OP_UNIQUE_CARRIER' in df.columns and 'route_id' in df.columns:
        df['carrier_route_id'] = df['OP_UNIQUE_CARRIER'] + '_' + df['route_id']
        
    if 'incident' in df.columns:
        # Time-decayed risk rates (half-life of 180 days to prioritize recent events)
        hl = 180
        if 'ORIGIN' in df.columns:
            df['airport_risk_rate'] = compute_historical_rates(df, 'ORIGIN', 'incident', decay_halflife_days=hl)
        if 'OP_UNIQUE_CARRIER' in df.columns:
            df['carrier_risk_rate'] = compute_historical_rates(df, 'OP_UNIQUE_CARRIER', 'incident', decay_halflife_days=hl)
        if 'route_id' in df.columns:
            df['route_risk_rate'] = compute_historical_rates(df, 'route_id', 'incident', decay_halflife_days=hl)
        if 'carrier_route_id' in df.columns:
            df['carrier_route_risk_rate'] = compute_historical_rates(df, 'carrier_route_id', 'incident', decay_halflife_days=hl)
            
    # Fill NAs in risk rates
    for col in ['airport_risk_rate', 'carrier_risk_rate', 'route_risk_rate', 'carrier_route_risk_rate']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # 2. NOAA Weather Severity Features
    # Wind gust / crosswind component estimation (assume typical 45-degree angle fallback)
    if 'wspd' in df.columns:
        df['crosswind_component_kt'] = df['wspd'] * 0.707
    else:
        df['crosswind_component_kt'] = 0.0
        
    # Populate NOAA severity indicators (if not present, default to 0/False)
    if 'icing_pirep_count_24h' not in df.columns:
        df['icing_pirep_count_24h'] = 0.0
    if 'sigmet_active' not in df.columns:
        df['sigmet_active'] = 0
    if 'convective_activity' not in df.columns:
        # Infer from precip rate or set to 0
        if 'prcp' in df.columns:
            df['convective_activity'] = (df['prcp'] > 5.0).astype(int)
        else:
            df['convective_activity'] = 0
            
    # Compute Adverse Weather Index
    # Components: Crosswind, Icing, Precipitation, Convective
    # Normalize components roughly to a 0-1 scale to sum them
    crosswind_score = (df['crosswind_component_kt'] / 30.0).clip(upper=1.0) # 30kt is severe
    icing_score = (df['icing_pirep_count_24h'] / 5.0).clip(upper=1.0)
    precip_score = 0.0
    if 'prcp' in df.columns:
        precip_score = (df['prcp'] / 10.0).clip(upper=1.0) # 10mm/hr is severe
    convective_score = df['convective_activity']
    
    df['adverse_weather_index'] = crosswind_score + icing_score + precip_score + convective_score

    return df