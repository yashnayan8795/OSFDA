"""
Upload schemas for the Manual Test feature.

Single source of truth: the validator AND the on-screen format instructions
both read from the same dict, so they can't disagree.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class UploadSchema:
    problem_id: str
    display_name: str
    required: list[str] = field(default_factory=list)
    optional: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    target: str | None = None
    notes: str = ""

    @property
    def all_columns(self) -> list[str]:
        return list(self.required) + list(self.optional)


SEVERITY_SCHEMA = UploadSchema(
    problem_id="severity",
    display_name="Problem A — Severity Classification",
    required=[
        "ACN",
        "Time_Date",
        "Place_Locale Reference",
        "Environment_Flight Conditions",
        "Aircraft 1_Flight Phase",
        "Person 1_Function",
        "Person 1_Qualification",
    ],
    optional=[
        "Place_State Reference",
        "Place_Altitude.AGL.Single Value",
        "Environment_Weather Elements / Visibility",
        "Aircraft 1_Make Model Name",
        "Aircraft 1_Mission",
        "Person 1_Experience.Flight Crew.Total",
        "Person 1_Experience.Flight Crew.Type",
        "Events_Anomaly",
    ],
    dtypes={
        "ACN": "string",
        "Time_Date": "string",
        "Place_Altitude.AGL.Single Value": "Int64",
        "Person 1_Experience.Flight Crew.Total": "Int64",
        "Person 1_Experience.Flight Crew.Type": "Int64",
    },
    target="Severity_Label",
    notes=(
        "ASRS-format incident reports. The model expects pre-circumstance fields "
        "(date, location, flight phase, crew qualification). Post-incident fields "
        "like *Detector* or *Resolution* are excluded to avoid leakage. "
        "`Time_Date` should be YYYYMM format (e.g. 201806). Include a "
        "`Severity_Label` column (0=None, 1=Minor, 2=Serious, 3=Fatal) if you want "
        "evaluation metrics; otherwise predictions are shown without ground truth."
    ),
)


PREFLIGHT_SCHEMA = UploadSchema(
    problem_id="preflight",
    display_name="Problem C — Pre-Flight Risk",
    required=[
        "FL_DATE",
        "OP_UNIQUE_CARRIER",
        "ORIGIN",
        "DEST",
        "CRS_DEP_TIME",
        "DISTANCE",
    ],
    optional=[
        "TAIL_NUM",
        "OP_CARRIER_FL_NUM",
        "CRS_ARR_TIME",
        "CRS_ELAPSED_TIME",
        "wx_temp_c",
        "wx_dwpt_c",
        "wx_rhum",
        "wx_wspd",
        "wx_wdir",
        "wx_pres",
        "wx_coco",
        "wx_prcp",
    ],
    dtypes={
        "FL_DATE": "datetime64[ns]",
        "CRS_DEP_TIME": "Int64",
        "CRS_ARR_TIME": "Int64",
        "CRS_ELAPSED_TIME": "Int64",
        "DISTANCE": "float64",
    },
    target="incident",
    notes=(
        "BTS On-Time-format flight records, optionally joined with weather "
        "(columns prefixed `wx_`). Missing weather is fine — the pipeline fills "
        "with training-set medians. `CRS_DEP_TIME` is HHMM integer (e.g. 1430 = 2:30 PM). "
        "Include an `incident` column (0/1) if you want evaluation metrics. "
        "Risk-rate features (airport / carrier / route) default to training means if absent."
    ),
)


SCHEMAS: dict[str, UploadSchema] = {
    "severity": SEVERITY_SCHEMA,
    "preflight": PREFLIGHT_SCHEMA,
}

UNSUPPORTED_PROBLEMS: dict[str, str] = {
    "category":  "Problem B — Category Classification",
    "discovery": "Problem D — Emerging Risk Discovery",
    "graph":     "Problem E — Factor Extraction",
}


def get_schema(problem_id: str) -> UploadSchema | None:
    return SCHEMAS.get(problem_id)
