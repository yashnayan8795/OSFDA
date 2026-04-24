"""
Tests for Leakage Audit — Feature Whitelist Gating
"""
import pytest
import pandas as pd
from src.data.leakage_audit import (
    validate_no_leakage,
    ALWAYS_EXCLUDED,
    POST_INCIDENT_COLUMNS,
)
from src.utils.config import load_feature_whitelist


@pytest.fixture
def whitelist():
    return load_feature_whitelist()


class TestLeakageAudit:
    def test_target_columns_always_flagged(self, whitelist):
        features = ["Time_Date", "Events.5_Result"]
        is_clean, leaks = validate_no_leakage(features, problem="A", whitelist=whitelist)
        assert not is_clean
        assert any("TARGET SOURCE" in l for l in leaks)

    def test_post_incident_flagged_for_problem_a(self, whitelist):
        features = ["Time_Date", "Events_Anomaly"]
        is_clean, leaks = validate_no_leakage(features, problem="A", whitelist=whitelist)
        assert not is_clean

    def test_post_incident_allowed_for_problem_b(self, whitelist):
        features = ["Time_Date", "Events_Anomaly"]
        is_clean, leaks = validate_no_leakage(features, problem="B", whitelist=whitelist)
        # Events_Anomaly is not a target source, so allowed in B
        assert is_clean or not any("TARGET SOURCE" in l for l in leaks)

    def test_clean_features_pass(self, whitelist):
        features = ["Time_Date", "Place.1_State Reference", "Aircraft 1_Make"]
        is_clean, leaks = validate_no_leakage(features, problem="A", whitelist=whitelist)
        assert is_clean
        assert leaks == []

    def test_assessments_column_flagged(self, whitelist):
        features = ["Time_Date", "Assessments.1_Primary Problem"]
        is_clean, leaks = validate_no_leakage(features, problem="A", whitelist=whitelist)
        assert not is_clean

    def test_narrative_flagged_for_problem_a(self, whitelist):
        features = ["Time_Date", "Report 1_Narrative"]
        is_clean, leaks = validate_no_leakage(features, problem="A", whitelist=whitelist)
        assert not is_clean
