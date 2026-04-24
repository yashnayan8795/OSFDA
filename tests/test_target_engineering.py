"""
Tests for Target Engineering — Severity Rubric & Category Taxonomy
"""
import pytest
import pandas as pd
import numpy as np
from src.data.target_engineering import (
    calculate_severity,
    apply_severity_rubric,
    validate_severity_distribution,
    rubric_hash,
    map_categories,
    map_primary_category,
)
from src.utils.config import load_category_taxonomy


# ---- Severity Rubric Tests ----

def _make_row(**kwargs):
    """Helper to create a minimal Series that mimics ASRS data."""
    defaults = {
        "Events.5_Result": "",
        "Events.1_Miss Distance": "",
        "Events.2_Were Passengers Involved In Event": "",
        "Component.3_Problem": "",
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


class TestSeverityRubric:
    def test_critical_destroyed(self):
        row = _make_row(**{"Events.5_Result": "Aircraft Destroyed"})
        assert calculate_severity(row) == 3

    def test_critical_fatal(self):
        row = _make_row(**{"Injury_Type": "Fatal"})
        assert calculate_severity(row) == 3

    def test_critical_evacuation(self):
        row = _make_row(**{"Events.5_Result": "Evacuation performed"})
        assert calculate_severity(row) == 3

    def test_substantial_damage(self):
        row = _make_row(**{"Events.5_Result": "Substantial damage to wing"})
        assert calculate_severity(row) == 2

    def test_substantial_emergency(self):
        row = _make_row(**{"Events.5_Result": "Emergency landing"})
        assert calculate_severity(row) == 2

    def test_moderate_minor_damage(self):
        row = _make_row(**{"Events.5_Result": "Minor damage to stabilizer"})
        assert calculate_severity(row) == 1

    def test_moderate_close_miss(self):
        row = _make_row(**{"Events.1_Miss Distance": "100-200 ft"})
        assert calculate_severity(row) == 1

    def test_minor_routine(self):
        row = _make_row(**{"Events.5_Result": "No damage reported"})
        assert calculate_severity(row) == 0

    def test_minor_empty(self):
        row = _make_row()
        assert calculate_severity(row) == 0

    def test_rubric_hash_stability(self):
        h1 = rubric_hash()
        h2 = rubric_hash()
        assert h1 == h2, "Rubric hash should be deterministic"
        assert len(h1) == 16


class TestSeverityDistribution:
    def test_validation_passes_expected(self):
        df = pd.DataFrame({"severity_level": [0]*45 + [1]*25 + [2]*20 + [3]*10})
        result = validate_severity_distribution(df)
        assert result["is_valid"]
        assert len(result["warnings"]) == 0

    def test_validation_too_many_critical(self):
        df = pd.DataFrame({"severity_level": [0]*20 + [1]*20 + [2]*20 + [3]*40})
        result = validate_severity_distribution(df)
        assert not result["is_valid"]
        assert any("too loose" in w for w in result["warnings"])


# ---- Category Taxonomy Tests ----

class TestCategoryTaxonomy:
    @pytest.fixture
    def taxonomy(self):
        return {
            "source_field": "Assessments_Contributing Factors / Situations",
            "categories": {
                "Flight_Operations": {"keywords": ["Human Factors", "Procedure"]},
                "Equipment_System": {"keywords": ["Aircraft", "Equipment/Tooling"]},
                "Environment": {"keywords": ["Weather", "Turbulence"]},
            },
        }

    def test_single_match(self, taxonomy):
        result = map_categories("Human Factors", taxonomy)
        assert "Flight_Operations" in result

    def test_multi_match(self, taxonomy):
        result = map_categories("Human Factors ; Weather", taxonomy)
        assert "Flight_Operations" in result
        assert "Environment" in result

    def test_no_match(self, taxonomy):
        result = map_categories("Something Unrelated", taxonomy)
        assert len(result) == 0

    def test_nan_input(self, taxonomy):
        result = map_categories(np.nan, taxonomy)
        assert result == []

    def test_primary_mapping(self, taxonomy):
        assert map_primary_category("Human Factors", taxonomy) == "Flight_Operations"
        assert map_primary_category("Unknown Issue", taxonomy) == "Other"
        assert map_primary_category(np.nan, taxonomy) == "Other"
