import pandas as pd
import numpy as np
from typing import List, Optional

def compute_historical_rates(
    df: pd.DataFrame,
    group_col: str,
    target_col: str = 'incident',
) -> pd.Series:
    """
    Compute rolling historical incident rates for a given entity.
    Avoids leakage using shift(1).
    """
    df = df.sort_values([group_col, 'FL_DATE'])

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