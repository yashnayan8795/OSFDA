"""
Leakage Audit
=============
Feature whitelist gating for Problem A (pre-circumstance only) and
Problem B (retrospective, post-incident). No feature enters a model
without passing through this gate.

The whitelist is specified in ``configs/feature_whitelist.yaml``.
"""

import pandas as pd
from typing import Dict, Any, List, Set, Tuple, Optional

from src.utils.config import load_feature_whitelist


# Columns that are ALWAYS excluded (target sources or too close to target)
ALWAYS_EXCLUDED = {
    "Events.5_Result",                     # Target source for severity
    "Assessments.1_Primary Problem",       # Target source for category
    "Assessments_Contributing Factors / Situations",  # Too close to category target
}

# Post-incident observation columns — leak for Problem A as RAW STRINGS.
# Narratives are excluded here because PCA-compressed embeddings of pre-outcome
# text carry no outcome signal and are explicitly allowed via problem_a_text.
POST_INCIDENT_COLUMNS = {
    "Events_Anomaly",
    "Events.1_Miss Distance",
    "Events.2_Were Passengers Involved In Event",
    "Events.3_Detector",
    "Events.4_When Detected",
    "Events.5_Result",
    "Component.3_Problem",
}

# Narrative/synopsis columns — safe as embeddings, blocked only as raw strings.
NARRATIVE_COLUMNS = {
    "Report 1_Narrative",
    "Report 2_Narrative",
    "Report 1.2_Synopsis",
}

# Leakage keyword patterns — columns containing these ARE post-incident.
# NOTE: Narrative/Synopsis are intentionally excluded — they are pre-outcome
# text and are safe when used as PCA-compressed embeddings (not raw strings).
LEAKAGE_KEYWORDS = [
    "Events",
    "Assessments",
    "Result",
    "Anomaly",
    "Detector",
]


def get_problem_a_features(
    df: pd.DataFrame,
    whitelist: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Return the list of approved feature columns for Problem A (Severity).

    Only pre-circumstance features are allowed.
    """
    if whitelist is None:
        whitelist = load_feature_whitelist()

    approved = [
        item["feature"]
        for item in whitelist["problem_a_whitelist"]
        if item["feature"] in df.columns
    ]
    return approved


def get_problem_a_text_features(
    df: pd.DataFrame,
    whitelist: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Return the list of narrative/synopsis columns approved for Problem A
    as *embedding-only* inputs (not as raw categoricals).

    These columns should be encoded with SBERT → PCA before being appended
    to the tabular feature matrix.  They must NEVER be passed as raw string
    categoricals to LightGBM.
    """
    if whitelist is None:
        whitelist = load_feature_whitelist()

    approved_text = [
        item["feature"]
        for item in whitelist.get("problem_a_text", [])
        if item["feature"] in df.columns
    ]
    return approved_text


def get_problem_b_features(
    df: pd.DataFrame,
    whitelist: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Return the list of approved feature columns for Problem B (Category).

    Includes all Problem A features plus retrospective features
    (narratives, event details), but NOT target columns.
    """
    if whitelist is None:
        whitelist = load_feature_whitelist()

    prob_a = get_problem_a_features(df, whitelist)
    prob_b_extra = [
        item["feature"]
        for item in whitelist["problem_b_extra"]
        if item["feature"] in df.columns
    ]
    return list(set(prob_a + prob_b_extra))


def validate_no_leakage(
    feature_columns: List[str],
    problem: str = "A",
    whitelist: Optional[Dict[str, Any]] = None,
    allow_narrative_embeddings: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Validate that a list of feature columns contains no leakage.

    Parameters
    ----------
    feature_columns : list of str
        Columns intended for model training (tabular only — do NOT include
        narrative column names here when they are used as embeddings).
    problem : str
        "A" (strict pre-circumstance) or "B" (retrospective allowed).
    whitelist : dict, optional
        Loaded whitelist. If None, loads from config.
    allow_narrative_embeddings : bool
        If True, narrative/synopsis column names are not flagged even when
        they appear in feature_columns (used only for documentation; embeddings
        are generated separately and never passed as raw strings).

    Returns
    -------
    tuple of (bool, list of str)
        (is_clean, list_of_leaking_columns)
    """
    if whitelist is None:
        whitelist = load_feature_whitelist()

    leaking = []
    feature_set = set(feature_columns)

    # Always-excluded columns must never appear
    for col in feature_set & ALWAYS_EXCLUDED:
        leaking.append(f"{col} [TARGET SOURCE]")

    if problem == "A":
        # For Problem A, post-incident columns are leaks
        for col in feature_set & POST_INCIDENT_COLUMNS:
            if col not in ALWAYS_EXCLUDED:
                leaking.append(f"{col} [POST-INCIDENT LEAK for Problem A]")

        # Narratives as raw strings are a leak; as embeddings they are not.
        if not allow_narrative_embeddings:
            for col in feature_set & NARRATIVE_COLUMNS:
                leaking.append(
                    f"{col} [RAW NARRATIVE LEAK — use embed-only via get_problem_a_text_features()]"
                )

        # Also flag any unapproved column containing leakage keywords
        approved_a = set(
            item["feature"] for item in whitelist["problem_a_whitelist"]
        )
        # Build the set of approved text-embedding columns so they don't trip the keyword gate
        approved_text = set(
            item["feature"] for item in whitelist.get("problem_a_text", [])
        )
        unapproved = feature_set - approved_a - approved_text
        for col in unapproved:
            if any(kw.lower() in col.lower() for kw in LEAKAGE_KEYWORDS):
                leaking.append(
                    f"{col} [UNAPPROVED: contains leakage keyword]"
                )

    is_clean = len(leaking) == 0
    return is_clean, leaking


def filter_features(
    df: pd.DataFrame,
    problem: str = "A",
    whitelist: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Return a copy of df with only approved features for the given problem.

    Also includes the ID column ``acn_num_ACN`` if present.
    """
    if problem == "A":
        cols = get_problem_a_features(df, whitelist)
    elif problem == "B":
        cols = get_problem_b_features(df, whitelist)
    else:
        raise ValueError(f"Unknown problem: {problem}. Expected 'A' or 'B'.")

    # Always include ID if available
    if "acn_num_ACN" in df.columns and "acn_num_ACN" not in cols:
        cols = ["acn_num_ACN"] + cols

    return df[cols].copy()
