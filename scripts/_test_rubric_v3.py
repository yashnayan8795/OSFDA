"""Test new rubric."""
import pandas as pd
from src.data.loader import load_raw_data, load_main_config
from src.data.target_engineering import _parse_miss_distance

config = load_main_config()
df = load_raw_data(config)

def calculate_severity_v3(row):
    result = str(row.get("Events.5_Result", "")).lower()
    anomaly = str(row.get("Events_Anomaly", "")).lower()
    miss_str = str(row.get("Events.1_Miss Distance", "")).strip()
    component = str(row.get("Component.3_Problem", "")).lower()

    # Level 3: Critical
    # physical injury, regained aircraft control, aircraft damaged + critical equipment + fire
    crit_result = ["physical injury", "regained aircraft control"]
    if any(kw in result for kw in crit_result): return 3
    if "cftt / cfit" in anomaly: return 3
    if "aircraft damaged" in result and "equipment problem critical" in anomaly and "smoke / fire / fumes" in anomaly: return 3

    # Level 2: Substantial
    # landed in emergency condition (demoted from L3), aircraft damaged, rejected takeoff,
    # smoke / fire / fumes, conflict nmac
    sub_result = ["landed in emergency condition", "aircraft damaged", "rejected takeoff"]
    sub_anomaly = ["smoke / fire / fumes", "conflict nmac"]
    if any(kw in result for kw in sub_result): return 2
    if any(kw in anomaly for kw in sub_anomaly): return 2
    # Component failure + critical equipment problem
    if "failed" in component and "equipment problem critical" in anomaly: return 2

    # Level 1: Moderate
    # diverted (demoted from L2), returned to departure airport (demoted from L2),
    # took evasive action, executed go around, missed approach, landed as precaution
    mod_result = ["diverted", "returned to departure airport", "took evasive action", "executed go around", "missed approach", "landed as precaution"]
    mod_anomaly = ["equipment problem critical", "fuel issue", "conflict airborne conflict", "wake vortex encounter"]
    if any(kw in result for kw in mod_result): return 1
    if any(kw in anomaly for kw in mod_anomaly): return 1
    min_dist = _parse_miss_distance(miss_str)
    if min_dist is not None and min_dist < 500: return 1
    if "failed" in component: return 1

    return 0

df['sev_v3'] = df.apply(calculate_severity_v3, axis=1)
dist = df['sev_v3'].value_counts(normalize=True).sort_index() * 100
print("=== NEW RUBRIC V3 DISTRIBUTION ===")
print(dist)
