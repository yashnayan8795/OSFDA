"""
Temporal Feature Extraction
============================
Time-based features from the YYYYMM-encoded Time_Date field and
chronological train/val/test split logic.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from ``Time_Date`` (YYYYMM integer).

    Creates:
        - ``year``: integer year
        - ``month``: integer month (1-12)
        - ``quarter``: quarter (1-4)
        - ``time_of_day_bucket``: from ``Time.1_Local Time Of Day`` if available
        - ``month_sin``, ``month_cos``: cyclical month encoding
    """
    df = df.copy()

    # Year and month
    if "year" not in df.columns:
        df["year"] = df["Time_Date"] // 100
    if "month" not in df.columns:
        df["month"] = df["Time_Date"] % 100

    # Quarter
    df["quarter"] = ((df["month"] - 1) // 3) + 1

    # Cyclical encoding of month (for models that benefit from continuity)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Time of day bucketing (if the column exists)
    tod_col = "Time.1_Local Time Of Day"
    if tod_col in df.columns:
        df["time_of_day_bucket"] = df[tod_col].fillna("Unknown")
    else:
        df["time_of_day_bucket"] = "Unknown"

    return df


def create_temporal_split(
    df: pd.DataFrame,
    train_end_year: int = 2017,
    val_year: int = 2018,
    test_start_year: int = 2019,
) -> pd.DataFrame:
    """
    Create chronological train/val/test split based on ``year`` column.

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``year`` column (or ``Time_Date`` to derive it).
    train_end_year : int
        Last year included in training set (inclusive).
    val_year : int
        Year used for validation.
    test_start_year : int
        First year of the test set (through end of data).

    Returns
    -------
    pd.DataFrame
        Copy with ``split`` column added: 'train', 'val', or 'test'.
    """
    df = df.copy()

    if "year" not in df.columns:
        df["year"] = df["Time_Date"] // 100

    def assign_split(year):
        if year <= train_end_year:
            return "train"
        elif year == val_year:
            return "val"
        else:
            return "test"

    df["split"] = df["year"].apply(assign_split)
    return df


def validate_temporal_split(df: pd.DataFrame) -> Dict:
    """
    Validate the temporal split has no overlap and report statistics.

    Returns
    -------
    dict
        Contains split sizes, year ranges, and overlap check.
    """
    result = {}
    for split_name in ["train", "val", "test"]:
        subset = df[df["split"] == split_name]
        result[split_name] = {
            "count": len(subset),
            "pct": round(len(subset) / len(df) * 100, 1),
            "year_min": int(subset["year"].min()) if len(subset) > 0 else None,
            "year_max": int(subset["year"].max()) if len(subset) > 0 else None,
        }

    # Overlap check
    train_years = set(df[df["split"] == "train"]["year"].unique())
    val_years = set(df[df["split"] == "val"]["year"].unique())
    test_years = set(df[df["split"] == "test"]["year"].unique())

    result["overlap_train_val"] = list(train_years & val_years)
    result["overlap_train_test"] = list(train_years & test_years)
    result["overlap_val_test"] = list(val_years & test_years)
    result["is_valid"] = (
        len(result["overlap_train_val"]) == 0
        and len(result["overlap_train_test"]) == 0
        and len(result["overlap_val_test"]) == 0
    )

    return result


def get_split_data(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Split data into train/val/test dicts of (X, y).

    Returns
    -------
    dict
        Keys: 'train', 'val', 'test'.
        Values: (X_df, y_series) tuples.
    """
    splits = {}
    for split_name in ["train", "val", "test"]:
        mask = df["split"] == split_name
        X = df.loc[mask, feature_cols].copy()
        y = df.loc[mask, target_col].copy()
        splits[split_name] = (X, y)
    return splits
