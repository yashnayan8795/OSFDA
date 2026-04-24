"""
ASRS Dataset Loader
====================
Handles downloading from Hugging Face, column inventory, parquet I/O,
and the canonical data loading interface used by all downstream modules.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from src.utils.config import (
    load_main_config,
    resolve_path,
    PROJECT_ROOT,
)


def download_asrs_dataset(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Download ASRS dataset from Hugging Face and return as DataFrame.

    Parameters
    ----------
    config : dict, optional
        Main config dict. If None, loads from configs/main_config.yaml.

    Returns
    -------
    pd.DataFrame
        Full ASRS dataset.
    """
    from datasets import load_dataset

    if config is None:
        config = load_main_config()

    hf_name = config["data"]["hf_dataset"]
    print(f"Downloading dataset: {hf_name}")
    dataset = load_dataset(hf_name)
    df = dataset["train"].to_pandas()
    print(f"Downloaded {len(df)} records with {len(df.columns)} columns.")
    return df


def save_raw_data(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Path:
    """
    Save raw DataFrame to parquet at the configured path.

    Returns the absolute path to the saved file.
    """
    if config is None:
        config = load_main_config()

    save_path = resolve_path(config["paths"]["raw_data"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(save_path, index=False)
    print(f"Raw data saved to {save_path}")
    return save_path


def load_raw_data(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Load the raw ASRS parquet file.

    If not found, downloads it first.
    """
    if config is None:
        config = load_main_config()

    raw_path = resolve_path(config["paths"]["raw_data"])
    if not raw_path.exists():
        print(f"Raw data not found at {raw_path}. Downloading...")
        df = download_asrs_dataset(config)
        save_raw_data(df, config)
        return df

    df = pd.read_parquet(raw_path)
    print(f"Loaded {len(df)} records from {raw_path}")
    return df


def column_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a detailed inventory of all columns.

    Returns
    -------
    pd.DataFrame
        Inventory with columns: column, dtype, missing_pct, unique_count, samples
    """
    rows = []
    for col in df.columns:
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_pct": round(df[col].isna().mean() * 100, 2),
                "unique_count": df[col].nunique(),
                "samples": df[col].dropna().unique()[:5].tolist(),
            }
        )
    inv = pd.DataFrame(rows).sort_values("missing_pct", ascending=False)
    return inv


def parse_time_date(df: pd.DataFrame, time_col: str = "Time_Date") -> pd.DataFrame:
    """
    Parse the YYYYMM-encoded Time_Date field into year and month columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `time_col`.
    time_col : str
        Name of the time column (default: ``Time_Date``).

    Returns
    -------
    pd.DataFrame
        Copy of df with ``year`` and ``month`` columns added.
    """
    df = df.copy()
    # Time_Date is an integer encoded as YYYYMM
    df["year"] = df[time_col] // 100
    df["month"] = df[time_col] % 100
    return df
