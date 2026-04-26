"""Run Phase 1: Data Foundation."""
from src.utils.config import load_main_config, resolve_path, set_seeds
from src.data.loader import load_raw_data, column_inventory, parse_time_date
from src.data.target_engineering import (
    apply_severity_rubric, validate_severity_distribution,
    apply_category_taxonomy, rubric_hash,
)
from src.data.leakage_audit import get_problem_a_features, validate_no_leakage
from src.utils.config import load_feature_whitelist
from src.features.temporal import create_temporal_split, validate_temporal_split
import pandas as pd

set_seeds()
config = load_main_config()

# Load data
df = load_raw_data(config)

# Severity
df = apply_severity_rubric(df)
val = validate_severity_distribution(df)
print("Severity distribution:", val["distribution"])
print("Valid:", val["is_valid"])
for w in val["warnings"]:
    print("WARNING:", w)
print("Rubric hash:", rubric_hash())

# Save severity targets
sev_path = resolve_path("data/processed/severity_targets.parquet")
sev_path.parent.mkdir(parents=True, exist_ok=True)
df[["acn_num_ACN", "severity_level"]].to_parquet(sev_path, index=False)

# Category
df, cat_matrix = apply_category_taxonomy(df)
print("\nCategory counts:")
print(cat_matrix.sum())
avg_labels = cat_matrix.sum(axis=1).mean()
zero = (cat_matrix.sum(axis=1) == 0).sum()
print(f"Avg labels/report: {avg_labels:.2f}")
print(f"Zero labels: {zero} ({zero/len(df)*100:.1f}%)")
print("Primary category distribution:")
print(df["primary_category"].value_counts())

cat_path = resolve_path("data/processed/category_targets.parquet")
cat_out = pd.concat([df[["acn_num_ACN", "primary_category"]], cat_matrix], axis=1)
cat_out.to_parquet(cat_path, index=False)

# Leakage
whitelist = load_feature_whitelist()
prob_a_feats = get_problem_a_features(df, whitelist)
is_clean, leaks = validate_no_leakage(prob_a_feats, problem="A", whitelist=whitelist)
print(f"\nProblem A features available: {len(prob_a_feats)}")
print(f"Leakage check passed: {is_clean}")
if not is_clean:
    for l in leaks:
        print(f"  LEAK: {l}")

# Temporal split
df = parse_time_date(df)
df = create_temporal_split(df)
split_info = validate_temporal_split(df)
print("\nTemporal split:")
for s in ["train", "val", "test"]:
    info = split_info[s]
    count = info["count"]
    pct = info["pct"]
    ymin = info["year_min"]
    ymax = info["year_max"]
    print(f"  {s}: {count} ({pct}%) — years {ymin}-{ymax}")
print(f"Valid (no overlap): {split_info['is_valid']}")

split_path = resolve_path("data/processed/temporal_splits.parquet")
df[["acn_num_ACN", "year", "month", "split"]].to_parquet(split_path, index=False)
print("\nPhase 1 complete!")
