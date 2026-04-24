"""
Tests for Category Model — Taxonomy, Threshold Tuning, Prediction Shape
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from src.data.target_engineering import (
    map_categories, map_primary_category, apply_category_taxonomy,
)
from src.utils.config import load_category_taxonomy
from src.models.category import tune_thresholds


# ─────────────────────────────────────────────────────────────────────────────
# Taxonomy v2 Tests (real keyword values)
# ─────────────────────────────────────────────────────────────────────────────

class TestTaxonomyV2:
    @pytest.fixture
    def taxonomy(self):
        return load_category_taxonomy()

    def test_flight_ops_human_factors(self, taxonomy):
        result = map_categories("Human Factors", taxonomy)
        assert "Flight_Operations" in result

    def test_flight_ops_procedure(self, taxonomy):
        result = map_categories("Procedure", taxonomy)
        assert "Flight_Operations" in result

    def test_equipment_aircraft(self, taxonomy):
        result = map_categories("Aircraft", taxonomy)
        assert "Equipment_System" in result

    def test_equipment_mel(self, taxonomy):
        result = map_categories("MEL", taxonomy)
        assert "Equipment_System" in result

    def test_atc_communication(self, taxonomy):
        result = map_categories("ATC Equipment / Nav Facility / Buildings", taxonomy)
        assert "ATC_Communication" in result

    def test_atc_airspace_structure(self, taxonomy):
        result = map_categories("Airspace Structure", taxonomy)
        assert "ATC_Communication" in result

    def test_atc_airport(self, taxonomy):
        result = map_categories("Airport", taxonomy)
        assert "ATC_Communication" in result

    def test_environment_weather(self, taxonomy):
        result = map_categories("Weather", taxonomy)
        assert "Environment" in result

    def test_environment_non_weather(self, taxonomy):
        result = map_categories("Environment - Non Weather Related", taxonomy)
        assert "Environment" in result

    def test_airspace_chart(self, taxonomy):
        result = map_categories("Chart Or Publication", taxonomy)
        assert "Airspace_Navigation" in result

    def test_multi_label(self, taxonomy):
        result = map_categories("Human Factors; Aircraft; Weather", taxonomy)
        assert "Flight_Operations" in result
        assert "Equipment_System" in result
        assert "Environment" in result

    def test_no_labels(self, taxonomy):
        result = map_categories("Something totally random", taxonomy)
        assert len(result) == 0

    def test_nan_returns_empty(self, taxonomy):
        result = map_categories(np.nan, taxonomy)
        assert result == []

    def test_primary_mapping_human_factors(self, taxonomy):
        assert map_primary_category("Human Factors", taxonomy) == "Flight_Operations"

    def test_primary_mapping_aircraft(self, taxonomy):
        assert map_primary_category("Aircraft", taxonomy) == "Equipment_System"

    def test_primary_mapping_weather(self, taxonomy):
        assert map_primary_category("Weather", taxonomy) == "Environment"

    def test_primary_mapping_atc(self, taxonomy):
        assert map_primary_category("ATC Equipment / Nav Facility / Buildings", taxonomy) == "ATC_Communication"

    def test_primary_mapping_unknown(self, taxonomy):
        assert map_primary_category("Something Unknown", taxonomy) == "Other"

    def test_primary_mapping_nan(self, taxonomy):
        assert map_primary_category(np.nan, taxonomy) == "Other"


# ─────────────────────────────────────────────────────────────────────────────
# Threshold Tuning Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestThresholdTuning:
    def _make_binary_probs(self, n=100, n_pos=30, seed=42):
        rng = np.random.RandomState(seed)
        y = np.zeros(n, dtype=int)
        y[:n_pos] = 1
        # Probs correlated with y, but noisy
        probs = rng.beta(2, 5, n)
        probs[y == 1] += 0.4
        probs = np.clip(probs, 0, 1)
        return probs, y

    def test_thresholds_in_range(self):
        probs_a, y_a = self._make_binary_probs()
        probs_b, y_b = self._make_binary_probs(seed=99)
        label_names = ["LabelA", "LabelB"]
        val_probs = {"LabelA": probs_a, "LabelB": probs_b}
        y_val = pd.DataFrame({"LabelA": y_a, "LabelB": y_b})
        thresholds = tune_thresholds(val_probs, y_val, label_names)
        for label in label_names:
            assert 0.0 < thresholds[label] < 1.0

    def test_zero_positive_class_defaults_to_half(self):
        probs = np.zeros(100)
        y_all_zero = np.zeros(100, dtype=int)
        y_val = pd.DataFrame({"EmptyLabel": y_all_zero})
        thresholds = tune_thresholds({"EmptyLabel": probs}, y_val, ["EmptyLabel"])
        assert thresholds["EmptyLabel"] == 0.5

    def test_threshold_improves_over_default(self):
        """Tuned threshold should give F1 >= F1 at default 0.5."""
        from sklearn.metrics import f1_score
        probs_a, y_a = self._make_binary_probs(n_pos=10)
        y_val = pd.DataFrame({"A": y_a})
        thresholds = tune_thresholds({"A": probs_a}, y_val, ["A"])
        tuned_f1 = f1_score(y_a, (probs_a >= thresholds["A"]).astype(int), zero_division=0)
        default_f1 = f1_score(y_a, (probs_a >= 0.5).astype(int), zero_division=0)
        assert tuned_f1 >= default_f1
