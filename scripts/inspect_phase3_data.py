"""Inspect category targets and narrative columns before Phase 3."""
import pandas as pd
import numpy as np

cat = pd.read_parquet('data/processed/category_targets.parquet')
sev = pd.read_parquet('data/processed/severity_targets.parquet')
splits = pd.read_parquet('data/processed/temporal_splits.parquet')
raw = pd.read_parquet('data/raw/asrs_full.parquet')

print("=== Category Targets ===")
print(f"Shape: {cat.shape}")
cat_labels = [c for c in cat.columns if c not in ['acn_num_ACN', 'primary_category']]
print(f"Label columns: {cat_labels}")
print("\nLabel counts:")
print(cat[cat_labels].sum())
avg_labels = cat[cat_labels].sum(axis=1).mean()
zero_rows = (cat[cat_labels].sum(axis=1) == 0).sum()
print(f"\nAvg labels/row: {avg_labels:.2f}")
print(f"Zero-label rows: {zero_rows} ({zero_rows/len(cat)*100:.1f}%)")
print("\nPrimary category dist:")
print(cat['primary_category'].value_counts())

# Narrative columns
print("\n=== Narrative Columns ===")
narrative_cols = [c for c in raw.columns if 'narrative' in c.lower() or 'synopsis' in c.lower()]
for col in narrative_cols:
    missing = raw[col].isna().mean()*100
    has_content = raw[col].dropna()
    avg_len = has_content.str.len().mean() if len(has_content) else 0
    print(f"  {col}: missing={missing:.1f}%, avg_len={avg_len:.0f} chars")

# Split-level label distribution
print("\n=== Per-Split Label Counts ===")
cat_with_split = cat.merge(splits[['acn_num_ACN', 'split']], on='acn_num_ACN')
for sp in ['train', 'val', 'test']:
    sub = cat_with_split[cat_with_split['split'] == sp]
    print(f"\n{sp.upper()} ({len(sub)} rows):")
    for col in cat_labels:
        n = sub[col].sum()
        print(f"  {col}: {n} ({n/len(sub)*100:.1f}%)")

# Sample a narrative with its category
print("\n=== Sample Narratives with Categories ===")
merged = cat.merge(splits[['acn_num_ACN', 'split']], on='acn_num_ACN')
merged = merged.merge(raw[['acn_num_ACN', 'Report 1_Narrative', 'Report 1.2_Synopsis']], on='acn_num_ACN')
for label in cat_labels[:3]:
    sample = merged[merged[label] == 1].sample(1, random_state=42)
    acn = sample['acn_num_ACN'].values[0]
    narr = str(sample['Report 1_Narrative'].values[0])[:200]
    print(f"\n--- {label} ---")
    print(f"ACN: {acn}")
    print(f"Narrative snippet: {narr}...")
