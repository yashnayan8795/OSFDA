# OSFDA — Data Directory

This directory holds all dataset files used by the OSFDA research pipeline.

---

## What is tracked by Git vs kept locally

### Git repository tracks:
- `data/README.md` — this file
- `data/metadata/` — data manifests, schema descriptions, column inventory
- Small intentional result files: `data/processed/emerging_risks.csv`, `data/processed/factor_graph.json`, `data/processed/optuna.db`

### Local working copy only (git-ignored):
- `data/raw/` — all raw source datasets (large binaries, download from external sources)
- `data/interim/` — intermediate transformed datasets
- `data/processed/shared/`, `data/processed/problem_a/`, `data/processed/problem_b/`, `data/processed/problem_c/` — model-ready datasets and embeddings

See `data/metadata/data_manifest.md` for a complete file-by-file inventory with reproduction instructions.

---

## Directory Semantics

`
data/
|-- raw/          Original externally sourced datasets - no OSFDA transformations applied.
|   |-- asrs/     NASA ASRS aviation safety reports (HuggingFace download)
|   |-- ntsb/     NTSB accident database + original source archives
|   |   |-- source/   avall.mdb, avall.zip - original NTSB downloads
|   |-- bts/      BTS Airline On-Time Performance data
|   |   |-- annual/   Canonical annual parquets (bts_20??.parquet) - used by pipeline
|   |   |-- monthly/  Monthly partitions (verified record-level equivalent to annual)
|   |   |-- source/   Original downloaded archives (test_bts.zip)
|   |-- weather/  NOAA weather data (if separately cached)
|
|-- interim/      Intermediate joins, sampling, and enrichment - not final model input.
|   |-- problem_c/  Case-control dataset and NOAA-enriched intermediate files
|
|-- processed/    Final model-ready datasets and shared reusable outputs.
|   |-- shared/       Used across multiple problems
|   |   |-- embeddings/  SBERT sentence embeddings (emb_train/val/test.npy)
|   |-- problem_a/    Severity classification targets
|   |-- problem_b/    Category classification targets
|   |-- problem_c/    Pre-flight risk feature dataset
|   |-- (result files stay here - emerging_risks.csv, factor_graph.json, etc.)
|
|-- metadata/     Tracked data manifests and schema descriptions.
`

---

## Data Sources

### NASA ASRS (Aviation Safety Reporting System)
- **Source:** HuggingFace dataset `elihoole/asrs-aviation-reports`
- **Format:** Parquet
- **Local path:** `data/raw/asrs/asrs_full.parquet`
- **Download:** Run `notebooks/shared/00_data_acquisition.ipynb`
- **Used by:** All problems (A, B, D, E)

### NTSB Accident Database
- **Source:** NTSB Aviation Accident Database - original MDB format in `data/raw/ntsb/source/`
- **Format:** Parquet (re-acquired via API)
- **Local path:** `data/raw/ntsb/ntsb_accidents.parquet`
- **Download:** Run `notebooks/problem_c/05a_ntsb_acquisition.ipynb`
- **Used by:** Problem C only

### BTS Airline On-Time Performance
- **Source:** BTS TranStats (transtats.bts.gov)
- **Format:** Parquet (annual files are canonical)
- **Local path:** `data/raw/bts/annual/bts_20{18,19,20}.parquet`
- **Download:** Run `notebooks/problem_c/05b_bts_flights.ipynb`
- **Used by:** Problem C only
- **Note:** 36 monthly files in `data/raw/bts/monthly/` are record-level equivalent to annual files (SHA-256 verified). Monthly files are retained for reference but not used by the production pipeline.

### NOAA Weather Data
- **Source:** NOAA API (Meteostat / NOAA GHCND)
- **Format:** Embedded in `data/interim/problem_c/preflight_weather_enriched.parquet`
- **Note:** No standalone weather raw file - NOAA API query is done during enrichment notebook.

---

## Data Lineage

### Shared (Problems A, B, D, E)

```
NASA ASRS (HuggingFace)
    |
    v  notebooks/shared/00_data_acquisition.ipynb
data/raw/asrs/asrs_full.parquet
    |
    v  scripts/run_phase1.py
data/processed/problem_a/severity_targets.parquet
data/processed/problem_b/category_targets.parquet
data/processed/shared/temporal_splits.parquet
    |
    v  scripts/run_phase3.py (Tier 2)
data/processed/shared/embeddings/emb_{train,val,test}.npy
    |
    v  scripts/run_phase4.py  --> Problem D outputs (data/processed/)
    v  scripts/run_phase5.py  --> Problem E outputs (data/processed/)
```

### Problem C (Pre-flight Risk)

```
NTSB accidents + BTS flights (annual)
    |
    v  notebooks/problem_c/05c_case_control.ipynb
data/interim/problem_c/preflight_casecontrol.parquet
    |
    v  notebooks/problem_c/05d_weather_enrichment.ipynb  (NOAA API)
data/interim/problem_c/preflight_weather_enriched.parquet
    |
    v  notebooks/problem_c/05e_preflight_features.ipynb
data/processed/problem_c/preflight_features_final.parquet
    |
    v  scripts/run_phase6.py
models/preflight_lgbm_calibrated.joblib
```

---

## BTS Annual vs Monthly Files

The 36 monthly BTS parquet files (`bts_YYYY_MM.parquet`) are **record-level equivalent** to the 3 annual files (`bts_20??.parquet`). This was verified by SHA-256 hash comparison of all rows sorted by `[FL_DATE, OP_UNIQUE_CARRIER, OP_CARRIER_FL_NUM, ORIGIN, DEST]`.

The production pipeline uses **annual files only** via the glob `data/raw/bts/annual/bts_20??.parquet`.

---

## Git Status Summary

| Path | Git Status |
|------|-----------|
| `data/README.md` | TRACKED |
| `data/metadata/` | TRACKED |
| `data/processed/emerging_risks.csv` | TRACKED |
| `data/processed/factor_graph.json` | TRACKED |
| `data/processed/optuna.db` | TRACKED |
| All other `data/**/*.parquet` | LOCAL ONLY |
| All `data/**/*.npy` | LOCAL ONLY |
| All `data/**/*.mdb` | LOCAL ONLY |
| All `data/**/*.zip` | LOCAL ONLY |
| Most `data/processed/*.csv` and `*.json` | LOCAL ONLY |
