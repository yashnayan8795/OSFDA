"""
Rubric Phrase Scanner — Narrative Safety Gate
=============================================
Scans ASRS narrative/synopsis columns for phrases that directly reveal
severity labels or rubric classification criteria.

These phrases come from post-incident outcome fields (Events.5_Result,
Events_Anomaly, Component.3_Problem) and must NOT appear in model
training features — even indirectly via narrative embeddings.

Usage
-----
    from src.data.rubric_scanner import scan_narratives, prepare_safe_narratives

    report = scan_narratives(df, text_cols=["Report 1_Narrative", "Report 1.2_Synopsis"])
    df = prepare_safe_narratives(df, text_cols, mask=True)
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Rubric-derived leak phrases — these come directly from the labeling rubric.
# Any of these appearing in narrative text would allow the model to infer
# severity labels without actually learning signal from pre-incident context.
# ---------------------------------------------------------------------------
RUBRIC_LEVEL3_PHRASES = [
    "physical injury",
    "regained aircraft control",
    "cftt",
    "cfit",
    "controlled flight into terrain",
]

RUBRIC_LEVEL2_PHRASES = [
    "landed in emergency condition",
    "aircraft damaged",
    "rejected takeoff",
    "smoke / fire / fumes",
    "smoke/fire/fumes",
    "smoke fire fumes",
    "conflict nmac",
    "near midair collision",
    "near mid-air collision",
]

RUBRIC_LEVEL1_PHRASES = [
    "returned to departure airport",
    "took evasive action",
    "executed go around",
    "execute go-around",
    "missed approach",
    "landed as precaution",
    "work refused",
    "equipment problem critical",
    "wake vortex encounter",
    "passenger misconduct",
    "fuel issue",
    "conflict airborne conflict",
]

RUBRIC_COMPONENT_PHRASES = [
    "component failed",
    "component malfunctioning",
]

# Combined list of all rubric-leaking phrases (ordered by specificity)
ALL_RUBRIC_PHRASES: List[str] = (
    RUBRIC_LEVEL3_PHRASES
    + RUBRIC_LEVEL2_PHRASES
    + RUBRIC_LEVEL1_PHRASES
    + RUBRIC_COMPONENT_PHRASES
)

# Compiled patterns (case-insensitive)
_COMPILED_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (phrase, re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE))
    for phrase in ALL_RUBRIC_PHRASES
]

MASK_TOKEN = "[SEVERITY_MASKED]"
MASK_THRESHOLD_PCT = 1.0  # Mask if phrase appears in >1% of documents


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def scan_narratives(
    df: pd.DataFrame,
    text_cols: Optional[List[str]] = None,
    phrases: Optional[List[str]] = None,
) -> Dict:
    """
    Scan narrative columns for rubric-leaking phrases.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    text_cols : list of str, optional
        Columns to scan. Defaults to standard ASRS narrative columns.
    phrases : list of str, optional
        Phrases to check. Defaults to ALL_RUBRIC_PHRASES.

    Returns
    -------
    dict with:
        - 'phrase_stats': {phrase: {count, pct, should_mask}}
        - 'total_docs': int
        - 'phrases_to_mask': list of phrases exceeding threshold
        - 'summary': str — human-readable summary
    """
    if text_cols is None:
        text_cols = ["Report 1_Narrative", "Report 1.2_Synopsis"]
    if phrases is None:
        phrases = ALL_RUBRIC_PHRASES

    text_cols = [c for c in text_cols if c in df.columns]
    if not text_cols:
        return {
            "phrase_stats": {},
            "total_docs": 0,
            "phrases_to_mask": [],
            "summary": "WARNING: No text columns found.",
        }

    # Combine all text for scanning
    combined = df[text_cols].fillna("").apply(lambda row: " ".join(row), axis=1).str.lower()
    total = len(combined)

    phrase_stats = {}
    phrases_to_mask = []

    for phrase in phrases:
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        matches = combined.str.contains(pattern, regex=True)
        count = int(matches.sum())
        pct = 100.0 * count / total if total > 0 else 0.0
        should_mask = pct > MASK_THRESHOLD_PCT
        phrase_stats[phrase] = {
            "count": count,
            "pct": round(pct, 3),
            "should_mask": should_mask,
        }
        if should_mask:
            phrases_to_mask.append(phrase)

    summary_lines = [
        f"Narrative rubric scan — {total} documents, {len(text_cols)} text columns",
        f"Phrases exceeding {MASK_THRESHOLD_PCT}% threshold (will be masked): {len(phrases_to_mask)}",
    ]
    for p in phrases_to_mask:
        s = phrase_stats[p]
        summary_lines.append(f"  MASK '{p}': {s['count']} docs ({s['pct']:.2f}%)")
    below = [p for p in phrases if not phrase_stats[p]["should_mask"] and phrase_stats[p]["count"] > 0]
    if below:
        summary_lines.append(f"Phrases below threshold (logged only): {len(below)}")
        for p in below:
            s = phrase_stats[p]
            summary_lines.append(f"  OK   '{p}': {s['count']} docs ({s['pct']:.2f}%)")

    return {
        "phrase_stats": phrase_stats,
        "total_docs": total,
        "phrases_to_mask": phrases_to_mask,
        "summary": "\n".join(summary_lines),
    }


def mask_rubric_phrases(
    text: str,
    phrases_to_mask: Optional[List[str]] = None,
    mask_token: str = MASK_TOKEN,
) -> str:
    """
    Replace rubric-leaking phrases in a single text string.

    Parameters
    ----------
    text : str
    phrases_to_mask : list of str
        If None, uses ALL_RUBRIC_PHRASES for maximum safety.
    mask_token : str
        Replacement string.

    Returns
    -------
    str with phrases replaced by mask_token.
    """
    if not isinstance(text, str) or not text.strip():
        return text or ""
    if phrases_to_mask is None:
        phrases_to_mask = ALL_RUBRIC_PHRASES
    for phrase in phrases_to_mask:
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        text = pattern.sub(mask_token, text)
    return text


def prepare_safe_narratives(
    df: pd.DataFrame,
    text_cols: Optional[List[str]] = None,
    phrases: Optional[List[str]] = None,
    mask: bool = True,
    output_suffix: str = "_safe",
) -> pd.DataFrame:
    """
    Scan narrative columns, then optionally mask rubric-leaking phrases.
    Adds masked columns with suffix (e.g. 'Report 1_Narrative_safe').

    Parameters
    ----------
    df : pd.DataFrame
    text_cols : list of str
    phrases : list of str, optional — phrases to mask. If None, auto-detected by scan.
    mask : bool — if True, mask detected phrases (default).
    output_suffix : str — suffix for masked column names.

    Returns
    -------
    (df_with_safe_cols, scan_report)
    """
    if text_cols is None:
        text_cols = ["Report 1_Narrative", "Report 1.2_Synopsis"]
    text_cols = [c for c in text_cols if c in df.columns]

    scan_report = scan_narratives(df, text_cols, phrases)
    phrases_to_mask = scan_report["phrases_to_mask"]

    df = df.copy()

    for col in text_cols:
        safe_col = f"{col}{output_suffix}"
        if mask and phrases_to_mask:
            df[safe_col] = df[col].fillna("").apply(
                lambda t: mask_rubric_phrases(t, phrases_to_mask)
            )
        else:
            # No masking needed (no phrases exceeded threshold) — use original
            df[safe_col] = df[col].fillna("")

    return df, scan_report


def print_scan_report(scan_report: Dict) -> None:
    """Print a scan report to stdout."""
    print("\n" + "=" * 60)
    print("  NARRATIVE RUBRIC SAFETY SCAN")
    print("=" * 60)
    print(scan_report["summary"])
    print("=" * 60 + "\n")
