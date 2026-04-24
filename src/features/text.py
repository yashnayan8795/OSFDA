"""
Text Preprocessing
===================
ASRS-specific text handling for narratives and synopses.
Handles redaction placeholders, boilerplate removal, and
preparation for both TF-IDF baselines and Sentence-BERT encoding.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Optional


# ASRS redaction placeholders
REDACTION_PATTERN = re.compile(r"\b(Z{2,4}|X{2,4})\b", re.IGNORECASE)

# Common ASRS boilerplate patterns
BOILERPLATE_PATTERNS = [
    re.compile(r"^ASRS\s+Report\s+Number.*?$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Callback\s*:.*?$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Synopsis\s*:", re.MULTILINE | re.IGNORECASE),
]


def clean_narrative(
    text: str,
    replace_redactions: bool = True,
    remove_boilerplate: bool = True,
) -> str:
    """
    Clean a single ASRS narrative string.

    Parameters
    ----------
    text : str
        Raw narrative text.
    replace_redactions : bool
        Replace ZZZ/XXX/ZZZZ/XXXX with [REDACTED].
    remove_boilerplate : bool
        Remove common ASRS header/footer patterns.

    Returns
    -------
    str
        Cleaned text.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    text = text.strip()

    # Replace redaction placeholders
    if replace_redactions:
        text = REDACTION_PATTERN.sub("[REDACTED]", text)

    # Remove boilerplate
    if remove_boilerplate:
        for pattern in BOILERPLATE_PATTERNS:
            text = pattern.sub("", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_narratives(
    df: pd.DataFrame,
    narrative_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Clean all narrative/synopsis columns in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain narrative columns.
    narrative_cols : list of str, optional
        Columns to clean. Defaults to standard ASRS narrative columns.

    Returns
    -------
    pd.DataFrame
        Copy with cleaned text columns (suffixed with ``_clean``).
    """
    if narrative_cols is None:
        narrative_cols = [
            "Report 1_Narrative",
            "Report 2_Narrative",
            "Report 1.2_Synopsis",
        ]

    df = df.copy()
    for col in narrative_cols:
        if col in df.columns:
            clean_col = f"{col}_clean"
            df[clean_col] = df[col].apply(clean_narrative)

    return df


def combine_text_fields(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    output_col: str = "combined_text",
) -> pd.DataFrame:
    """
    Combine multiple text fields into a single column for embedding.

    Uses cleaned versions if available, otherwise raw.
    """
    if cols is None:
        # Prefer cleaned versions
        cols = []
        for base in ["Report 1_Narrative", "Report 1.2_Synopsis"]:
            clean = f"{base}_clean"
            if clean in df.columns:
                cols.append(clean)
            elif base in df.columns:
                cols.append(base)

    df = df.copy()
    df[output_col] = df[cols].fillna("").agg(" ".join, axis=1).str.strip()
    return df


def count_redactions(text: str) -> int:
    """Count redaction placeholders in text."""
    if pd.isna(text) or not isinstance(text, str):
        return 0
    return len(REDACTION_PATTERN.findall(text))


def redaction_stats(df: pd.DataFrame, col: str = "Report 1_Narrative") -> pd.DataFrame:
    """
    Compute per-report redaction statistics.

    Returns DataFrame with redaction_count, token_count, redaction_pct.
    """
    df = df.copy()
    df["redaction_count"] = df[col].apply(count_redactions)
    df["token_count"] = df[col].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )
    df["redaction_pct"] = np.where(
        df["token_count"] > 0,
        (df["redaction_count"] / df["token_count"]) * 100,
        0,
    )
    return df[["redaction_count", "token_count", "redaction_pct"]]
