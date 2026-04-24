"""
Tests for Target Engineering — Severity Rubric v2 & Category Taxonomy
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
    _parse_miss_distance,
)


# ---- Helpers ----

def _make_row(**kwargs):
    """Create a minimal Series that mimics ASRS data."""
    defaults = {
        "Events.5_Result": "",
        "Events_Anomaly": "",
        "Events.1_Miss Distance": "",
        "Component.3_Problem": "",
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


# ---- Miss Distance Parsing ----

class TestMissDistanceParsing:
    def test_horizontal_vertical(self):
        assert _parse_miss_distance("Horizontal 100; Vertical 200") == 100

    def test_vertical_only(self):
        assert _parse_miss_distance("Vertical 300") == 300

    def test_zero_distance(self):
        assert _parse_miss_distance("Horizontal 0; Vertical 100") == 0

    def test_empty(self):
        assert _parse_miss_distance("") is None

    def test_nan(self):
        assert _parse_miss_distance("nan") is None


# ---- Severity Rubric v2 Tests ----

class TestSeverityRubric:
    def test_critical_emergency_landing(self):
        row = _make_row(**{"Events.5_Result": "Flight Crew Landed in Emergency Condition"})
        assert calculate_severity(row) == 3

    def test_critical_physical_injury(self):
        row = _make_row(**{"Events.5_Result": "General Physical Injury / Incapacitation"})
        assert calculate_severity(row) == 3

    def test_critical_regained_control(self):
        row = _make_row(**{"Events.5_Result": "Flight Crew Regained Aircraft Control"})
        assert calculate_severity(row) == 3

    def test_critical_cftt(self):
        row = _make_row(**{"Events_Anomaly": "Inflight Event / Encounter CFTT / CFIT"})
        assert calculate_severity(row) == 3

    def test_substantial_aircraft_damaged(self):
        row = _make_row(**{"Events.5_Result": "Aircraft Aircraft Damaged"})
        assert calculate_severity(row) == 2

    def test_substantial_diverted(self):
        row = _make_row(**{"Events.5_Result": "Flight Crew Diverted"})
        assert calculate_severity(row) == 2

    def test_substantial_equipment_critical(self):
        row = _make_row(**{"Events_Anomaly": "Aircraft Equipment Problem Critical"})
        assert calculate_severity(row) == 2

    def test_substantial_nmac(self):
        row = _make_row(**{"Events_Anomaly": "Conflict NMAC"})
        assert calculate_severity(row) == 2

    def test_substantial_component_failed(self):
        row = _make_row(**{"Component.3_Problem": "Failed"})
        assert calculate_severity(row) == 2

    def test_substantial_fire(self):
        row = _make_row(**{
            "Events_Anomaly": "Flight Deck / Cabin / Aircraft Event Smoke / Fire / Fumes / Odor"
        })
        assert calculate_severity(row) == 2

    def test_moderate_evasive_action(self):
        row = _make_row(**{"Events.5_Result": "Flight Crew Took Evasive Action"})
        assert calculate_severity(row) == 1

    def test_moderate_go_around(self):
        row = _make_row(**{"Events.5_Result": "Flight Crew Executed Go Around / Missed Approach"})
        assert calculate_severity(row) == 1

    def test_moderate_equipment_less_severe(self):
        row = _make_row(**{"Events_Anomaly": "Aircraft Equipment Problem Less Severe"})
        assert calculate_severity(row) == 1

    def test_moderate_airborne_conflict(self):
        row = _make_row(**{"Events_Anomaly": "Conflict Airborne Conflict"})
        assert calculate_severity(row) == 1

    def test_moderate_close_miss(self):
        row = _make_row(**{"Events.1_Miss Distance": "Horizontal 100; Vertical 200"})
        assert calculate_severity(row) == 1

    def test_moderate_malfunction(self):
        row = _make_row(**{"Component.3_Problem": "Malfunctioning"})
        assert calculate_severity(row) == 1

    def test_minor_none_reported(self):
        row = _make_row(**{"Events.5_Result": "General None Reported / Taken"})
        assert calculate_severity(row) == 0

    def test_minor_reoriented(self):
        row = _make_row(**{"Events.5_Result": "Flight Crew Became Reoriented"})
        assert calculate_severity(row) == 0

    def test_minor_empty(self):
        row = _make_row()
        assert calculate_severity(row) == 0

    def test_rubric_hash_stability(self):
        h1 = rubric_hash()
        h2 = rubric_hash()
        assert h1 == h2
        assert len(h1) == 16


class TestSeverityDistribution:
    def test_validation_passes_expected(self):
        df = pd.DataFrame({"severity_level": [0]*40 + [1]*30 + [2]*20 + [3]*10})
        result = validate_severity_distribution(df)
        assert result["is_valid"]

    def test_validation_too_many_critical(self):
        df = pd.DataFrame({"severity_level": [0]*20 + [1]*20 + [2]*20 + [3]*40})
        result = validate_severity_distribution(df)
        assert not result["is_valid"]

    def test_validation_too_many_minor(self):
        df = pd.DataFrame({"severity_level": [0]*80 + [1]*10 + [2]*5 + [3]*5})
        result = validate_severity_distribution(df)
        assert not result["is_valid"]


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
        assert len(map_categories("Something Unrelated", taxonomy)) == 0

    def test_nan_input(self, taxonomy):
        assert map_categories(np.nan, taxonomy) == []

    def test_primary_mapping(self, taxonomy):
        assert map_primary_category("Human Factors", taxonomy) == "Flight_Operations"
        assert map_primary_category("Unknown Issue", taxonomy) == "Other"
        assert map_primary_category(np.nan, taxonomy) == "Other"
