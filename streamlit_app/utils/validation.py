"""
Validation for manually uploaded test data.

Policy:
  - REQUIRED columns: strict — missing any fails validation.
  - OPTIONAL columns: lenient — missing ones are filled with NaN + warning.
  - Row cap: 10,000 — above that, fail with a clear message.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .schemas import UploadSchema

MAX_ROWS = 10_000


@dataclass
class ValidationResult:
    ok: bool
    df: pd.DataFrame | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def read_upload(uploaded_file) -> tuple[pd.DataFrame | None, str | None]:
    """Read a CSV or Parquet upload into a DataFrame. Returns (df, error_msg)."""
    name = (uploaded_file.name or "").lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith(".parquet") or name.endswith(".pq"):
            df = pd.read_parquet(uploaded_file)
        else:
            return None, f"Unsupported file type: `{uploaded_file.name}`. Use .csv or .parquet."
        return df, None
    except Exception as e:
        return None, f"Could not read `{uploaded_file.name}`: {e}"


def validate(df: pd.DataFrame, schema: UploadSchema) -> ValidationResult:
    """Run strict-required + lenient-optional validation against a schema."""
    errors: list[str] = []
    warnings: list[str] = []

    if len(df) > MAX_ROWS:
        errors.append(
            f"Upload has {len(df):,} rows; the limit is {MAX_ROWS:,}. "
            f"Trim or sample the file before uploading."
        )
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    if len(df) == 0:
        errors.append("Upload is empty (0 rows).")
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    missing_required = [c for c in schema.required if c not in df.columns]
    if missing_required:
        errors.append(
            "Missing required columns: "
            + ", ".join(f"`{c}`" for c in missing_required)
        )
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    df = df.copy()
    missing_optional = [c for c in schema.optional if c not in df.columns]
    if missing_optional:
        for c in missing_optional:
            df[c] = pd.NA
        warnings.append(
            f"{len(missing_optional)} optional column(s) missing — filled with NaN: "
            + ", ".join(f"`{c}`" for c in missing_optional[:6])
            + ("…" if len(missing_optional) > 6 else "")
        )

    for col, dtype in schema.dtypes.items():
        if col not in df.columns:
            continue
        try:
            if dtype.startswith("datetime"):
                df[col] = pd.to_datetime(df[col], errors="coerce")
            else:
                df[col] = df[col].astype(dtype, errors="ignore")
        except Exception as e:
            warnings.append(f"Could not coerce `{col}` to {dtype}: {e}")

    if schema.target and schema.target not in df.columns:
        warnings.append(
            f"No `{schema.target}` column found — predictions will be shown "
            f"without evaluation metrics."
        )

    known = set(schema.all_columns) | ({schema.target} if schema.target else set())
    extras = [c for c in df.columns if c not in known]
    if extras:
        warnings.append(
            f"{len(extras)} extra column(s) will be ignored by the model: "
            + ", ".join(f"`{c}`" for c in extras[:6])
            + ("…" if len(extras) > 6 else "")
        )

    return ValidationResult(ok=True, df=df, errors=errors, warnings=warnings)
