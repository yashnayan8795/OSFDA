"""
Target Engineering
==================
Deterministic severity rubric (Problem A) and multi-label category
taxonomy (Problem B) from physical outcome fields — NOT from
post-incident narrative keyword matching.

The rubric is versioned and specified in configs/severity_rubric_v1.yaml.
The taxonomy is specified in configs/category_taxonomy_v1.yaml.

RUBRIC v2.0 — Calibrated against actual ASRS field values:
  - Events.5_Result uses phrases like "Aircraft Aircraft Damaged",
    "Flight Crew Landed in Emergency Condition", etc.
  - Events_Anomaly uses phrases like "Aircraft Equipment Problem Critical",
    "Conflict NMAC", "Smoke / Fire / Fumes"
  - Events.1_Miss Distance uses "Horizontal N; Vertical N" format
  - Component.3_Problem uses "Failed", "Malfunctioning", "Design"
  - No dedicated injury columns — injury info is in Events.5_Result
"""

import hashlib
import inspect
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from src.utils.config import load_severity_rubric, load_category_taxonomy


# ---------------------------------------------------------------------------
# Problem A: Severity Rubric v2.0 (calibrated to real data)
# ---------------------------------------------------------------------------

def _parse_miss_distance(miss_str: str) -> Optional[float]:
    """
    Parse 'Horizontal N; Vertical N' into minimum distance in feet.
    Returns None if unparseable.
    """
    if not miss_str or miss_str in {"nan", ""}:
        return None
    import re
    numbers = re.findall(r"(\d+)", miss_str)
    if not numbers:
        return None
    # Return the minimum of all parsed distances
    return min(int(n) for n in numbers)


def calculate_severity(row: pd.Series) -> int:
    """
    Deterministic severity rubric v2.1.

    Derives ordinal severity (0–3) from physical outcome fields ONLY.
    These fields are POST-incident and must NEVER appear in the feature set.
    """
    result = str(row.get("Events.5_Result", "")).lower()
    anomaly = str(row.get("Events_Anomaly", "")).lower()
    miss_str = str(row.get("Events.1_Miss Distance", "")).strip()
    component = str(row.get("Component.3_Problem", "")).lower()

    # ---- Level 3: Critical ----
    crit_result = ["physical injury", "regained aircraft control"]
    if any(kw in result for kw in crit_result): return 3
    if "cftt / cfit" in anomaly: return 3
    if "aircraft damaged" in result and "equipment problem critical" in anomaly and "smoke / fire / fumes" in anomaly: return 3

    # ---- Level 2: Substantial ----
    sub_result = ["landed in emergency condition", "aircraft damaged", "rejected takeoff"]
    sub_anomaly = ["smoke / fire / fumes", "conflict nmac"]
    if any(kw in result for kw in sub_result): return 2
    if any(kw in anomaly for kw in sub_anomaly): return 2
    # Component failure + critical equipment problem
    if "failed" in component and "equipment problem critical" in anomaly: return 2

    # ---- Level 1: Moderate ----
    mod_result = ["diverted", "returned to departure airport", "took evasive action", "executed go around", "missed approach", "landed as precaution", "work refused"]
    mod_anomaly = ["equipment problem critical", "fuel issue", "conflict airborne conflict", "wake vortex encounter", "passenger misconduct"]
    if any(kw in result for kw in mod_result): return 1
    if any(kw in anomaly for kw in mod_anomaly): return 1
    min_dist = _parse_miss_distance(miss_str)
    if min_dist is not None and min_dist < 500: return 1
    if "failed" in component: return 1

    # ---- Level 0: Minor ----
    return 0


def apply_severity_rubric(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the severity rubric to the full dataset."""
    df = df.copy()
    df["severity_level"] = df.apply(calculate_severity, axis=1)
    return df


def validate_severity_distribution(
    df: pd.DataFrame,
    min_level3_pct: float = 2.0,
    max_level3_pct: float = 20.0,   # ASRS self-selection bias → more critical reports
    max_level0_pct: float = 70.0,
) -> Dict[str, Any]:
    """Validate severity distribution against expected ranges."""
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

    return {
        "distribution": dist.to_dict(),
        "is_valid": len(warnings) == 0,
        "warnings": warnings,
    }


def rubric_hash() -> str:
    """SHA-256 hash of calculate_severity source for versioning."""
    src = inspect.getsource(calculate_severity)
    return hashlib.sha256(src.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Problem B: Category Taxonomy
# ---------------------------------------------------------------------------

def map_categories(factors_str: str, taxonomy: Dict[str, Any]) -> List[str]:
    """Map semicolon-delimited contributing factors to multi-label categories."""
    if pd.isna(factors_str):
        return []
    factors = [f.strip().lower() for f in str(factors_str).split(";")]
    assigned = set()
    for cat, details in taxonomy["categories"].items():
        keywords = [k.lower() for k in details["keywords"]]
        if any(any(kw in factor for kw in keywords) for factor in factors):
            assigned.add(cat)
    return sorted(assigned)


def map_primary_category(primary_str: str, taxonomy: Dict[str, Any]) -> str:
    """Map Assessments.1_Primary Problem to a single primary category."""
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
    """Apply multi-label category taxonomy to the full dataset."""
    from sklearn.preprocessing import MultiLabelBinarizer

    if taxonomy is None:
        taxonomy = load_category_taxonomy()

    source_field = taxonomy["source_field"]
    df = df.copy()

    df["categories"] = df[source_field].apply(
        lambda x: map_categories(x, taxonomy)
    )

    if "Assessments.1_Primary Problem" in df.columns:
        df["primary_category"] = df["Assessments.1_Primary Problem"].apply(
            lambda x: map_primary_category(x, taxonomy)
        )
    else:
        df["primary_category"] = "Other"

    category_names = sorted(taxonomy["categories"].keys())
    mlb = MultiLabelBinarizer(classes=category_names)
    category_matrix = mlb.fit_transform(df["categories"])
    category_df = pd.DataFrame(category_matrix, columns=mlb.classes_, index=df.index)

    return df, category_df
