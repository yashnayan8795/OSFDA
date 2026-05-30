"""
    Feature Encoding
    =================
    Categorical and numeric encoding strategies for LightGBM/CatBoost
    with native categorical support, plus fallback encoding for
    scikit-learn baselines.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple


def identify_column_types(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    Classify columns into categorical, numeric, and high-missing groups.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.
    exclude_cols : list, optional
        Columns to skip (e.g., IDs, targets).

    Returns
    -------
    dict with keys:
        - 'categorical': list of object/category columns
        - 'numeric': list of numeric columns
        - 'high_missing': list of columns with >80% missing
        - 'medium_missing': list of columns with 50-80% missing
    """
    if exclude_cols is None:
        exclude_cols = []

    cols = [c for c in df.columns if c not in exclude_cols]
    missing = df[cols].isna().mean()

    result = {
        "categorical": [],
        "numeric": [],
        "high_missing": missing[missing > 0.80].index.tolist(),
        "medium_missing": missing[(missing > 0.50) & (missing <= 0.80)].index.tolist(),
    }

    for col in cols:
        if col in result["high_missing"]:
            continue  # Will be dropped
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            result["categorical"].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            result["numeric"].append(col)

    return result


def prepare_for_lgbm(
    df: pd.DataFrame,
    categorical_cols: List[str],
    numeric_cols: List[str],
    medium_missing_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Prepare DataFrame for LightGBM with native categorical support.

    - Converts categorical columns to ``category`` dtype.
    - Adds ``_is_missing`` indicator flags for medium-missing columns.
    - LightGBM handles NaN in categoricals natively.
    """
    df = df.copy()

    # Convert to category dtype for native LightGBM support
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Add missingness indicators for medium-missing columns
    if medium_missing_cols:
        for col in medium_missing_cols:
            if col in df.columns:
                df[f"{col}_is_missing"] = df[col].isna().astype(int)

    return df


def bucket_experience(
    df: pd.DataFrame,
    exp_col: str = "Person 1.5_Experience",
) -> pd.DataFrame:
    """
    Bucket flight experience into ordinal bands.

    Bands: 0-1000, 1000-5000, 5000-15000, 15000+
    """
    df = df.copy()
    if exp_col not in df.columns:
        return df

    # Try to parse numeric values from the column
    exp_numeric = pd.to_numeric(df[exp_col], errors="coerce")

    bins = [0, 1000, 5000, 15000, float("inf")]
    labels = ["low_0_1k", "mid_1k_5k", "high_5k_15k", "expert_15k+"]
    df["experience_bucket"] = pd.cut(
        exp_numeric, bins=bins, labels=labels, right=True
    )
    df["experience_bucket"] = df["experience_bucket"].astype("category")

    return df


def engineer_interaction_features(
    df: pd.DataFrame,
) -> tuple:
    """
    Engineer pre-flight interaction features from whitelisted Problem A columns.

    All source columns are pre-incident (from problem_a_whitelist).
    The interactions are string concatenations treated as new categoricals.

    Created features
    ----------------
    - ``phase_aircraft``         : Flight Phase × Aircraft Make/Model
    - ``experience_conditions``  : Experience bucket × Flight Conditions
    - ``crew_size_mission``      : Crew Size × Mission type

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns from problem_a_whitelist (already prepared).

    Returns
    -------
    (df_with_interactions, new_feature_names)
        df_with_interactions : pd.DataFrame with new columns added.
        new_feature_names    : list of str — names of the new feature columns.
    """
    df = df.copy()
    new_features = []

    def _safe_str(series: pd.Series) -> pd.Series:
        return series.fillna("unknown").astype(str).str.strip().str.lower()

    # 1. Flight Phase × Aircraft Make/Model
    phase_col = "Aircraft 1.9_Flight Phase"
    aircraft_col = "Aircraft 1.2_Make Model Name"
    if phase_col in df.columns and aircraft_col in df.columns:
        df["phase_aircraft"] = (
            _safe_str(df[phase_col]) + "__" + _safe_str(df[aircraft_col])
        ).astype("category")
        new_features.append("phase_aircraft")

    # 2. Experience Bucket × Flight Conditions
    exp_col = "experience_bucket"
    conditions_col = "Environment_Flight Conditions"
    if exp_col in df.columns and conditions_col in df.columns:
        df["experience_conditions"] = (
            _safe_str(df[exp_col]) + "__" + _safe_str(df[conditions_col])
        ).astype("category")
        new_features.append("experience_conditions")

    # 3. Crew Size × Mission Type
    crew_col = "Aircraft 1.4_Crew Size"
    mission_col = "Aircraft 1.7_Mission"
    if crew_col in df.columns and mission_col in df.columns:
        df["crew_size_mission"] = (
            _safe_str(df[crew_col]) + "__" + _safe_str(df[mission_col])
        ).astype("category")
        new_features.append("crew_size_mission")

    return df, new_features


def frequency_encode(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Frequency encoding for high-cardinality categoricals (baseline LogReg).

    Computes frequency from TRAIN only, applies to val/test.
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    for col in cols:
        if col not in train_df.columns:
            continue
        freq_map = train_df[col].value_counts(normalize=True).to_dict()
        suffix = f"{col}_freq"
        train_df[suffix] = train_df[col].map(freq_map).fillna(0)
        val_df[suffix] = val_df[col].map(freq_map).fillna(0)
        test_df[suffix] = test_df[col].map(freq_map).fillna(0)

    return train_df, val_df, test_df
