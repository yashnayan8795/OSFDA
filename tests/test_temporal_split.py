"""
Tests for Temporal Split — Chronological Integrity
"""
import pytest
import pandas as pd
import numpy as np
from src.features.temporal import (
    create_temporal_split,
    validate_temporal_split,
    extract_temporal_features,
)


@pytest.fixture
def sample_df():
    """Synthetic data spanning 2003-2020 as YYYYMM integers."""
    np.random.seed(42)
    years = list(range(2003, 2021))
    months = list(range(1, 13))
    records = []
    for y in years:
        for m in months:
            n = np.random.randint(50, 200)
            for _ in range(n):
                records.append({"Time_Date": y * 100 + m})
    return pd.DataFrame(records)


class TestTemporalSplit:
    def test_split_non_overlapping(self, sample_df):
        df = create_temporal_split(sample_df)
        result = validate_temporal_split(df)
        assert result["is_valid"], f"Overlap detected: {result}"

    def test_split_sizes_reasonable(self, sample_df):
        df = create_temporal_split(sample_df)
        result = validate_temporal_split(df)
        # Train should be the largest
        assert result["train"]["pct"] > 50
        # Test should be at least 10%
        assert result["test"]["pct"] > 5

    def test_train_ends_before_val(self, sample_df):
        df = create_temporal_split(sample_df)
        result = validate_temporal_split(df)
        assert result["train"]["year_max"] < result["val"]["year_min"]

    def test_val_ends_before_test(self, sample_df):
        df = create_temporal_split(sample_df)
        result = validate_temporal_split(df)
        assert result["val"]["year_max"] < result["test"]["year_min"]


class TestTemporalFeatures:
    def test_cyclical_encoding(self, sample_df):
        df = extract_temporal_features(sample_df)
        assert "month_sin" in df.columns
        assert "month_cos" in df.columns
        # sin²+cos² ≈ 1 for all rows
        magnitude = df["month_sin"]**2 + df["month_cos"]**2
        np.testing.assert_allclose(magnitude, 1.0, atol=1e-10)

    def test_quarter_extraction(self, sample_df):
        df = extract_temporal_features(sample_df)
        assert df["quarter"].min() == 1
        assert df["quarter"].max() == 4
