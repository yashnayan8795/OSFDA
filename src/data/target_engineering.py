"""
Target Engineering
==================
Deterministic severity rubric (Problem A) and multi-label category
taxonomy (Problem B) from physical outcome fields — NOT from
post-incident narrative keyword matching.

The rubric is versioned and specified in configs/severity_rubric_v1.yaml.
The taxonomy is specified in configs/category_taxonomy_v1.yaml.
"""

import hashlib
import inspect
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from src.utils.config import load_severity_rubric, load_category_taxonomy


# ---------------------------------------------------------------------------
# Problem A: Severity Rubric
# ---------------------------------------------------------------------------

def calculate_severity(row: pd.Series) -> int:
    """
    Deterministic severity rubric v1.0.

    Derives ordinal severity (0–3) from physical outcome fields ONLY.
    These fields are POST-incident and must NEVER appear in the feature set.

    Source fields used:
        - Events.5_Result          → physical outcome
        - Events.1_Miss Distance   → near-miss severity
        - Injury columns           → any column containing 'Injury'
        - Component.3_Problem      → equipment failure indicator

    Returns
    -------
    int
        Severity level: 0 (Minor), 1 (Moderate), 2 (Substantial), 3 (Critical)
    """
    # Normalize text fields
    result = str(row.get("Events.5_Result", "")).lower()
    miss_dist = str(row.get("Events.1_Miss Distance", "")).lower().strip()
    component_problem = str(row.get("Component.3_Problem", "")).lower()

    # Gather all injury-related columns
    injury_cols = [c for c in row.index if "injury" in c.lower()]
    injuries = [str(row.get(c, "")).lower() for c in injury_cols]
    all_injuries = " ".join(injuries)

    # Gather damage-related columns
    damage_cols = [c for c in row.index if "damage" in c.lower()]
    damages = [str(row.get(c, "")).lower() for c in damage_cols]
    all_damages = " ".join(damages)

    # Also check passenger involvement
    pax_involved = str(
        row.get("Events.2_Were Passengers Involved In Event", "")
    ).lower()

    # ---- Level 3: Critical ----
    # Aircraft destroyed, fatalities, evacuation, or loss of control
    critical_result_kw = ["destroyed", "evacuation", "loss of control"]
    if any(kw in result for kw in critical_result_kw):
        return 3
    if "fatal" in all_injuries:
        return 3
    if "destroyed" in all_damages:
        return 3

    # ---- Level 2: Substantial ----
    # Substantial damage, injuries, or emergency declared
    substantial_result_kw = ["substantial", "emergency"]
    if any(kw in result for kw in substantial_result_kw):
        return 2
    serious_injury_kw = ["serious", "minor injury", "hospitali"]
    if any(kw in all_injuries for kw in serious_injury_kw):
        return 2
    if "substantial" in all_damages:
        return 2
    # Emergency / component failure indicating serious degradation
    if "fire" in component_problem or "failure" in component_problem:
        return 2

    # ---- Level 1: Moderate ----
    # Minor damage or close near-miss
    close_miss_categories = {
        "less than 100 ft",
        "100-200 ft",
        "200-500 ft",
        "less than 100 feet",
        "100-200 feet",
        "200-500 feet",
    }
    if "minor" in result and "damage" in result:
        return 1
    if miss_dist in close_miss_categories:
        return 1
    if "minor" in all_damages:
        return 1
    # Near-miss with passengers
    if miss_dist and miss_dist not in {"nan", "", "none"} and pax_involved == "yes":
        return 1

    # ---- Level 0: Minor ----
    return 0


def apply_severity_rubric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the severity rubric to the full dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Full ASRS DataFrame.

    Returns
    -------
    pd.DataFrame
        Copy of df with ``severity_level`` column added.
    """
    df = df.copy()
    df["severity_level"] = df.apply(calculate_severity, axis=1)
    return df


def validate_severity_distribution(
    df: pd.DataFrame,
    min_level3_pct: float = 2.0,
    max_level3_pct: float = 15.0,
    max_level0_pct: float = 70.0,
) -> Dict[str, Any]:
    """
    Validate the severity distribution against expected ranges.

    Returns a dict with:
        distribution: per-level percentage
        is_valid: bool
        warnings: list of warning strings
    """
    dist = df["severity_level"].value_counts(normalize=True).sort_index() * 100
    warnings = []

    if dist.get(3, 0) > max_level3_pct:
        warnings.append(
            f"Level 3 (Critical) is {dist.get(3, 0):.1f}% — rubric may be too loose "
            f"(threshold: {max_level3_pct}%)"
        )
    if dist.get(3, 0) < min_level3_pct:
        warnings.append(
            f"Level 3 (Critical) is {dist.get(3, 0):.1f}% — may be too strict "
            f"(minimum expected: {min_level3_pct}%)"
        )
    if dist.get(0, 0) > max_level0_pct:
        warnings.append(
            f"Level 0 (Minor) is {dist.get(0, 0):.1f}% — rubric may be too strict "
            f"(threshold: {max_level0_pct}%)"
        )

    is_valid = len(warnings) == 0
    return {
        "distribution": dist.to_dict(),
        "is_valid": is_valid,
        "warnings": warnings,
    }


def rubric_hash() -> str:
    """
    Return SHA-256 hash of the ``calculate_severity`` function source.
    Enables rubric versioning — if the logic changes, the hash changes.
    """
    src = inspect.getsource(calculate_severity)
    return hashlib.sha256(src.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Problem B: Category Taxonomy
# ---------------------------------------------------------------------------

def map_categories(
    factors_str: str,
    taxonomy: Dict[str, Any],
) -> List[str]:
    """
    Map semicolon-delimited contributing factors to multi-label categories.

    Parameters
    ----------
    factors_str : str
        Semicolon-separated factor string from
        ``Assessments_Contributing Factors / Situations``.
    taxonomy : dict
        Loaded taxonomy spec (from ``category_taxonomy_v1.yaml``).

    Returns
    -------
    list of str
        Assigned category names.
    """
    if pd.isna(factors_str):
        return []

    factors = [f.strip().lower() for f in str(factors_str).split(";")]
    assigned = set()

    for cat, details in taxonomy["categories"].items():
        keywords = [k.lower() for k in details["keywords"]]
        if any(any(kw in factor for kw in keywords) for factor in factors):
            assigned.add(cat)

    return sorted(assigned)


def map_primary_category(
    primary_str: str,
    taxonomy: Dict[str, Any],
) -> str:
    """
    Map ``Assessments.1_Primary Problem`` to a single primary category.

    Returns 'Other' if no taxonomy match is found.
    """
    if pd.isna(primary_str):
        return "Other"

    primary = str(primary_str).lower()
    for cat, details in taxonomy["categories"].items():
        keywords = [k.lower() for k in details["keywords"]]
        if any(kw in primary for kw in keywords):
            return cat
    return "Other"


def apply_category_taxonomy(
    df: pd.DataFrame,
    taxonomy: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply multi-label category taxonomy to the full dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Full ASRS DataFrame.
    taxonomy : dict, optional
        Loaded taxonomy spec. If None, loads from config.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        - df copy with ``categories`` list column and ``primary_category`` column
        - Binary indicator DataFrame (n_samples × n_categories)
    """
    from sklearn.preprocessing import MultiLabelBinarizer

    if taxonomy is None:
        taxonomy = load_category_taxonomy()

    source_field = taxonomy["source_field"]
    df = df.copy()

    # Multi-label from contributing factors
    df["categories"] = df[source_field].apply(
        lambda x: map_categories(x, taxonomy)
    )

    # Primary category from Assessments.1_Primary Problem
    if "Assessments.1_Primary Problem" in df.columns:
        df["primary_category"] = df["Assessments.1_Primary Problem"].apply(
            lambda x: map_primary_category(x, taxonomy)
        )
    else:
        df["primary_category"] = "Other"

    # Binarize multi-labels
    category_names = sorted(taxonomy["categories"].keys())
    mlb = MultiLabelBinarizer(classes=category_names)
    category_matrix = mlb.fit_transform(df["categories"])
    category_df = pd.DataFrame(category_matrix, columns=mlb.classes_, index=df.index)

    return df, category_df
