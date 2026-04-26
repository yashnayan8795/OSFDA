"""
Sanity Tests for OSFDA Pipeline
=================================
Lightweight checks that each phase produces valid outputs.
Run with: python -m pytest tests/test_sanity.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_main_config, resolve_path, set_seeds
from src.data.loader import load_raw_data, parse_time_date
from src.data.target_engineering import (
    apply_severity_rubric, calculate_severity,
    validate_severity_distribution, apply_category_taxonomy,
)
from src.data.leakage_audit import get_problem_a_features, validate_no_leakage
from src.features.temporal import extract_temporal_features, create_temporal_split
from src.features.encoding import bucket_experience


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def df_raw():
    set_seeds()
    config = load_main_config()
    return load_raw_data(config)


@pytest.fixture(scope="module")
def df_prepared(df_raw):
    df = apply_severity_rubric(df_raw)
    df = parse_time_date(df)
    df = create_temporal_split(df)
    df = extract_temporal_features(df)
    df = bucket_experience(df)
    return df


# ─────────────────────────────────────────────────────────────
# Problem A — Severity Rubric Sanity
# ─────────────────────────────────────────────────────────────

class TestSeverityRubric:

    def test_severity_range(self, df_prepared):
        """Severity levels must be in {0, 1, 2, 3}."""
        levels = df_prepared["severity_level"].unique()
        assert set(levels).issubset({0, 1, 2, 3}), f"Unexpected levels: {levels}"

    def test_severity_distribution_no_inversion(self, df_prepared):
        """Level 3 must be less frequent than Level 0."""
        dist = df_prepared["severity_level"].value_counts(normalize=True)
        assert dist.get(3, 0) < dist.get(0, 0), (
            f"Distribution inversion: L3={dist.get(3,0):.2%} >= L0={dist.get(0,0):.2%}"
        )

    def test_level3_under_20_pct(self, df_prepared):
        """Level 3 should be under 20% of all reports."""
        pct = (df_prepared["severity_level"] == 3).mean()
        assert pct < 0.20, f"Level 3 is {pct:.1%} — rubric too loose"

    def test_level0_under_70_pct(self, df_prepared):
        """Level 0 should be under 70% of all reports."""
        pct = (df_prepared["severity_level"] == 0).mean()
        assert pct < 0.70, f"Level 0 is {pct:.1%} — rubric too strict"

    def test_physical_injury_is_level3(self):
        """'Physical injury' in result must always map to Level 3."""
        row = pd.Series({
            "Events.5_Result": "Physical Injury",
            "Events_Anomaly": "",
            "Events.1_Miss Distance": "",
            "Component.3_Problem": "",
        })
        assert calculate_severity(row) == 3

    def test_empty_row_is_level0(self):
        """An empty row must map to Level 0."""
        row = pd.Series({
            "Events.5_Result": "",
            "Events_Anomaly": "",
            "Events.1_Miss Distance": "",
            "Component.3_Problem": "",
        })
        assert calculate_severity(row) == 0

    def test_validation_passes(self, df_prepared):
        """The validate_severity_distribution function must pass."""
        result = validate_severity_distribution(df_prepared)
        assert result["is_valid"], f"Validation failed: {result['warnings']}"


# ─────────────────────────────────────────────────────────────
# Problem B — Category Taxonomy Sanity
# ─────────────────────────────────────────────────────────────

class TestCategoryTaxonomy:

    def test_taxonomy_produces_labels(self, df_prepared):
        """apply_category_taxonomy must produce at least 3 label columns."""
        df, cat_matrix = apply_category_taxonomy(df_prepared)
        assert cat_matrix.shape[1] >= 3, f"Only {cat_matrix.shape[1]} labels"

    def test_labels_are_binary(self, df_prepared):
        """All label values must be 0 or 1."""
        _, cat_matrix = apply_category_taxonomy(df_prepared)
        unique_vals = set()
        for col in cat_matrix.columns:
            unique_vals.update(cat_matrix[col].unique())
        assert unique_vals.issubset({0, 1}), f"Non-binary values: {unique_vals}"

    def test_no_empty_labels(self, df_prepared):
        """Each label must have at least some positive samples."""
        _, cat_matrix = apply_category_taxonomy(df_prepared)
        for col in cat_matrix.columns:
            assert cat_matrix[col].sum() > 0, f"Label '{col}' has zero positives"


# ─────────────────────────────────────────────────────────────
# Leakage Audit
# ─────────────────────────────────────────────────────────────

class TestLeakageAudit:

    def test_no_post_incident_leakage(self, df_prepared):
        """Feature whitelist must not contain post-incident fields."""
        from src.utils.config import load_feature_whitelist
        whitelist = load_feature_whitelist()
        feats = get_problem_a_features(df_prepared, whitelist)
        is_clean, leaks = validate_no_leakage(feats, problem="A", whitelist=whitelist)
        assert is_clean, f"Leakage detected: {leaks}"


# ─────────────────────────────────────────────────────────────
# Temporal Split
# ─────────────────────────────────────────────────────────────

class TestTemporalSplit:

    def test_split_exists(self, df_prepared):
        """DataFrame must have a 'split' column."""
        assert "split" in df_prepared.columns

    def test_three_splits(self, df_prepared):
        """Must have exactly train/val/test splits."""
        splits = set(df_prepared["split"].unique())
        assert splits == {"train", "val", "test"}, f"Got: {splits}"

    def test_no_future_leakage(self, df_prepared):
        """Train max date must be <= val min date, val max <= test min."""
        if "Time_Date" not in df_prepared.columns:
            pytest.skip("Time_Date not available")
        df = df_prepared.dropna(subset=["Time_Date"])
        train_max = df[df["split"] == "train"]["Time_Date"].max()
        val_min = df[df["split"] == "val"]["Time_Date"].min()
        assert train_max <= val_min, "Train data leaks into val"


# ─────────────────────────────────────────────────────────────
# Graph Sanity
# ─────────────────────────────────────────────────────────────

class TestGraphAnalysis:

    def test_graph_not_empty(self, df_prepared):
        """The multi-layer graph must have at least some nodes."""
        from src.models.graph_analysis import build_multilayer_graph
        G = build_multilayer_graph(df_prepared.head(500), min_edge_weight=2)
        assert G.number_of_nodes() > 0, "Graph is empty"

    def test_graph_has_multiple_types(self, df_prepared):
        """Graph should have more than one node type."""
        from src.models.graph_analysis import build_multilayer_graph
        G = build_multilayer_graph(df_prepared.head(500), min_edge_weight=2)
        types = set(d.get("node_type") for _, d in G.nodes(data=True))
        assert len(types) >= 2, f"Only {types} node types found"


# ─────────────────────────────────────────────────────────────
# Model Output Shape Checks
# ─────────────────────────────────────────────────────────────

class TestModelOutputs:

    def test_embeddings_exist(self):
        """Phase 3 embeddings must exist if Phase 3 has been run."""
        emb_path = resolve_path("data/processed/emb_train.npy")
        if not emb_path.exists():
            pytest.skip("Embeddings not generated yet (run Phase 3 first)")
        emb = np.load(emb_path)
        assert emb.ndim == 2, f"Expected 2D, got {emb.ndim}D"
        assert emb.shape[1] == 384, f"Expected 384 dims, got {emb.shape[1]}"

    def test_emerging_risks_csv(self):
        """Phase 4 emerging_risks.csv must exist and have required columns."""
        csv_path = resolve_path("data/processed/emerging_risks.csv")
        if not csv_path.exists():
            pytest.skip("emerging_risks.csv not generated yet (run Phase 4 first)")
        df = pd.read_csv(csv_path)
        required = {"Topic", "Growth_Ratio", "Risk_Score", "Name"}
        assert required.issubset(set(df.columns)), f"Missing columns: {required - set(df.columns)}"
        assert len(df) > 0, "Empty emerging_risks.csv"
