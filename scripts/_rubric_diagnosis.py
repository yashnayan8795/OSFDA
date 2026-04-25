"""Diagnostic script: understand what's inflating severity levels 2 and 3."""
import pandas as pd
from src.data.loader import load_raw_data, load_main_config
from src.data.target_engineering import apply_severity_rubric

config = load_main_config()
df = load_raw_data(config)
df = apply_severity_rubric(df)

# ── Level 3 breakdown ──
level3 = df[df['severity_level'] == 3]
print(f'=== LEVEL 3 TRIGGERS (n={len(level3)}) ===')
result3 = level3['Events.5_Result'].fillna('').str.lower()
for kw in ['landed in emergency condition', 'physical injury', 'regained aircraft control']:
    n = result3.str.contains(kw).sum()
    print(f'  Result "{kw}": {n} ({n/len(level3)*100:.1f}%)')
anomaly3 = level3['Events_Anomaly'].fillna('').str.lower()
n_cftt = anomaly3.str.contains('cftt / cfit').sum()
print(f'  Anomaly "cftt / cfit": {n_cftt} ({n_cftt/len(level3)*100:.1f}%)')
# combined fire+damage+critical
n_combo = ((result3.str.contains('aircraft damaged')) &
           (anomaly3.str.contains('equipment problem critical')) &
           (anomaly3.str.contains('smoke / fire / fumes'))).sum()
print(f'  Triple combo (damaged+critical+fire): {n_combo}')
print()

# ── Level 2 breakdown ──
level2 = df[df['severity_level'] == 2]
print(f'=== LEVEL 2 TRIGGERS (n={len(level2)}) ===')
result2 = level2['Events.5_Result'].fillna('').str.lower()
for kw in ['aircraft damaged', 'diverted', 'returned to departure', 'rejected takeoff']:
    n = result2.str.contains(kw).sum()
    print(f'  Result "{kw}": {n} ({n/len(level2)*100:.1f}%)')
anomaly2 = level2['Events_Anomaly'].fillna('').str.lower()
for kw in ['equipment problem critical', 'smoke / fire / fumes', 'conflict nmac', 'fuel issue']:
    n = anomaly2.str.contains(kw).sum()
    print(f'  Anomaly "{kw}": {n} ({n/len(level2)*100:.1f}%)')
comp2 = level2['Component.3_Problem'].fillna('').str.lower()
n_failed = comp2.str.contains('failed').sum()
print(f'  Component "failed": {n_failed} ({n_failed/len(level2)*100:.1f}%)')
print()

# ── Component.3_Problem global analysis ──
print('=== COMPONENT.3_PROBLEM GLOBAL ===')
comp_all = df['Component.3_Problem'].fillna('').str.lower()
print(f'Total "failed": {comp_all.str.contains("failed").sum()}')
print(f'Total "malfunctioning": {comp_all.str.contains("malfunctioning").sum()}')
print()
print('Component.3_Problem top 15 values:')
print(df['Component.3_Problem'].value_counts().head(15))
print()

# ── Mutual exclusivity: how many L2 are triggered ONLY by Component.3_Problem? ──
# Re-run rubric logic without component check
def severity_no_component(row):
    result = str(row.get("Events.5_Result", "")).lower()
    anomaly = str(row.get("Events_Anomaly", "")).lower()
    miss_str = str(row.get("Events.1_Miss Distance", "")).strip()
    crit_result = ["landed in emergency condition", "physical injury", "regained aircraft control"]
    if any(kw in result for kw in crit_result): return 3
    if "cftt / cfit" in anomaly: return 3
    if "aircraft damaged" in result and "equipment problem critical" in anomaly and "smoke / fire / fumes" in anomaly: return 3
    sub_result = ["aircraft damaged", "diverted", "returned to departure airport", "rejected takeoff"]
    sub_anomaly = ["equipment problem critical", "smoke / fire / fumes", "conflict nmac", "fuel issue"]
    if any(kw in result for kw in sub_result): return 2
    if any(kw in anomaly for kw in sub_anomaly): return 2
    # SKIP component check
    mod_result = ["took evasive action", "executed go around", "missed approach", "landed as precaution", "work refused"]
    mod_anomaly = ["equipment problem less severe", "conflict airborne conflict", "wake vortex encounter", "passenger misconduct"]
    if any(kw in result for kw in mod_result): return 1
    if any(kw in anomaly for kw in mod_anomaly): return 1
    import re
    numbers = re.findall(r"(\d+)", miss_str)
    if numbers:
        min_dist = min(int(n) for n in numbers)
        if min_dist < 500: return 1
    # SKIP component malfunctioning check
    return 0

df['sev_no_comp'] = df.apply(severity_no_component, axis=1)
print('=== DISTRIBUTION WITHOUT COMPONENT.3_PROBLEM ===')
print(df['sev_no_comp'].value_counts().sort_index())
print()
print('=== CURRENT DISTRIBUTION ===')
print(df['severity_level'].value_counts().sort_index())
print()

# How many rows shifted because of Component.3_Problem?
shifted = (df['severity_level'] != df['sev_no_comp']).sum()
print(f'Rows shifted by Component.3_Problem: {shifted}')

# Of the current Level 2 rows, how many would be Level 0 without component?
l2_mask = df['severity_level'] == 2
print(f'Current L2 rows: {l2_mask.sum()}')
print(f'  Would be L0 without component: {(df.loc[l2_mask, "sev_no_comp"] == 0).sum()}')
print(f'  Would be L1 without component: {(df.loc[l2_mask, "sev_no_comp"] == 1).sum()}')
print(f'  Stay L2 without component: {(df.loc[l2_mask, "sev_no_comp"] == 2).sum()}')
