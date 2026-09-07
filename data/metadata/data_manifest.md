# OSFDA Data Manifest
**Last updated:** 2026-09-07
**Purpose:** Complete inventory of all data files used by the OSFDA research pipeline.

---

## RAW DATA

### `data/raw/asrs/asrs_full.parquet`
| Field | Value |
|-------|-------|
| Logical name | NASA ASRS Aviation Safety Reports |
| Classification | RAW |
| Source | HuggingFace: `elihoole/asrs-aviation-reports` |
| Produced by | `notebooks/00_data_acquisition.ipynb` |
| Consumed by | Problems A, B, D, E (all ASRS-based problems) |
| Expected shape | ~38,000 rows x ~80 cols |
| Git status | LOCAL ONLY |
| Regeneration | Run `notebooks/00_data_acquisition.ipynb` (HuggingFace download, ~2 min) |

---

### `data/raw/ntsb/ntsb_accidents.parquet`
| Field | Value |
|-------|-------|
| Logical name | NTSB Aviation Accident Database |
| Classification | RAW |
| Source | NTSB Aviation Accident Database API |
| Produced by | `notebooks/05a_ntsb_acquisition.ipynb` |
| Consumed by | Problem C only — `notebooks/05c_case_control.ipynb`, `src/data/preflight.py` |
| Expected shape | ~4,704 rows x ~25 cols |
| Git status | LOCAL ONLY |
| Regeneration | Run `notebooks/05a_ntsb_acquisition.ipynb` |

### `data/raw/ntsb/source/avall.mdb`
| Field | Value |
|-------|-------|
| Logical name | NTSB Original MDB Source |
| Classification | ARCHIVE |
| Source | NTSB direct download |
| Produced by | External download |
| Consumed by | Not referenced by any pipeline code |
| Expected shape | ~520 MB (Microsoft Access database) |
| Git status | LOCAL ONLY |
| Regeneration | Download from NTSB website |

### `data/raw/ntsb/source/avall.zip`
| Field | Value |
|-------|-------|
| Logical name | NTSB Original ZIP Archive |
| Classification | ARCHIVE |
| Source | NTSB direct download |
| Produced by | External download |
| Consumed by | Not referenced by any pipeline code |
| Expected shape | ~90 MB |
| Git status | LOCAL ONLY |
| Regeneration | Download from NTSB website |

---

### `data/raw/bts/annual/bts_2018.parquet`
### `data/raw/bts/annual/bts_2019.parquet`
### `data/raw/bts/annual/bts_2020.parquet`
| Field | Value |
|-------|-------|
| Logical name | BTS Airline On-Time Performance (Annual) |
| Classification | RAW |
| Source | BTS TranStats (transtats.bts.gov) |
| Produced by | `notebooks/05b_bts_flights.ipynb` |
| Consumed by | Problem C — `src/data/preflight.py::load_preflight_raw()` |
| Expected shape | 2018: 7.2M rows; 2019: 7.4M rows; 2020: 4.7M rows; all 21 cols |
| Git status | LOCAL ONLY |
| Regeneration | Run `notebooks/05b_bts_flights.ipynb` |

### `data/raw/bts/monthly/bts_YYYY_MM.parquet` (36 files)
| Field | Value |
|-------|-------|
| Logical name | BTS Monthly Partitions |
| Classification | RAW |
| Source | BTS TranStats (monthly granularity) |
| Produced by | `notebooks/05b_bts_flights.ipynb` |
| Consumed by | **Not used by production pipeline** |
| Expected shape | ~0.5M–0.9M rows per month x 21 cols |
| Git status | LOCAL ONLY |
| Note | SHA-256 verified equivalent to annual files (bts_verify.py, 2026-09-06) |

### `data/raw/bts/source/test_bts.zip`
| Field | Value |
|-------|-------|
| Logical name | BTS Test Archive |
| Classification | ARCHIVE |
| Source | BTS TranStats download |
| Produced by | External download |
| Consumed by | Not referenced by any pipeline code |
| Expected shape | ~27 MB |
| Git status | LOCAL ONLY |
| Decision | Retained pending user confirmation |

---

## METADATA

### `data/metadata/column_inventory.csv`
| Field | Value |
|-------|-------|
| Logical name | ASRS Column Inventory |
| Classification | METADATA |
| Source | Generated from ASRS dataset |
| Produced by | `notebooks/00_data_acquisition.ipynb` (via `src/data/loader.py::column_inventory()`) |
| Consumed by | Repository reference only — not loaded by pipeline code |
| Expected shape | ~80 rows x 5 cols (column, dtype, missing_pct, unique_count, samples) |
| Git status | TRACKED (small, useful without downloading datasets) |
| Regeneration | Run `notebooks/00_data_acquisition.ipynb` |

### `data/metadata/data_manifest.md`
| Field | Value |
|-------|-------|
| Logical name | This file |
| Classification | METADATA / DOCUMENTATION |
| Git status | TRACKED |

---

## INTERIM DATA

### `data/interim/problem_c/preflight_casecontrol.parquet`
| Field | Value |
|-------|-------|
| Logical name | Problem C Case-Control Dataset |
| Classification | INTERIM |
| Source | NTSB accidents matched to BTS flights |
| Produced by | `notebooks/05c_case_control.ipynb` |
| Consumed by | `notebooks/05d_weather_enrichment.ipynb` |
| Expected shape | ~1,012,278 rows x 23 cols; incident rate 4.76% |
| Git status | LOCAL ONLY |
| Regeneration | Run `notebooks/05c_case_control.ipynb` (requires NTSB + BTS annual) |

### `data/interim/problem_c/preflight_weather_enriched.parquet`
| Field | Value |
|-------|-------|
| Logical name | Problem C Weather-Enriched Dataset |
| Classification | INTERIM |
| Source | Case-control + NOAA weather API |
| Produced by | `notebooks/05d_weather_enrichment.ipynb` |
| Consumed by | `notebooks/05e_preflight_features.ipynb` |
| Expected shape | ~1,012,278 rows x 31 cols |
| Git status | LOCAL ONLY |
| Regeneration | Run `notebooks/05d_weather_enrichment.ipynb` (requires NOAA API access) |

---

## PROCESSED DATA — SHARED

### `data/processed/shared/temporal_splits.parquet`
| Field | Value |
|-------|-------|
| Logical name | Temporal Train/Val/Test Split Assignments |
| Classification | PROCESSED |
| Source | ASRS time-based split (train: pre-2019, val: 2019, test: 2020+) |
| Produced by | `scripts/run_phase1.py` |
| Consumed by | Problems A, B — `streamlit_app/utils/loaders.py`, `streamlit_app/pages/2_Problem_B_Category.py`, `notebooks/01d_temporal_split_eda.ipynb`, `notebooks/06a_*`, `notebooks/06b_*` |
| Expected shape | ~38,000 rows x 4 cols (acn_num_ACN, year, month, split) |
| Git status | LOCAL ONLY |
| Regeneration | Run `scripts/run_phase1.py` |

### `data/processed/shared/embeddings/emb_train.npy`
### `data/processed/shared/embeddings/emb_val.npy`
### `data/processed/shared/embeddings/emb_test.npy`
| Field | Value |
|-------|-------|
| Logical name | SBERT Sentence Embeddings (all-MiniLM-L6-v2) |
| Classification | PROCESSED |
| Source | ASRS combined narrative text |
| Produced by | `scripts/run_phase3.py` (Tier 2) |
| Consumed by | `scripts/run_phase3.py` (Tier 3), `scripts/run_phase4.py`, `notebooks/02b_category_model.ipynb` |
| Expected shape | train: (28,000, 384); val: (5,600, 384); test: (4,400, 384) approx. |
| Git status | LOCAL ONLY |
| Regeneration | Run `scripts/run_phase3.py --tier 2` (~2h CPU or ~15min GPU) |
| Note | Model: `all-MiniLM-L6-v2` from `sentence-transformers`. Results are deterministic given the same model version. |

---

## PROCESSED DATA — PROBLEM A

### `data/processed/problem_a/severity_targets.parquet`
| Field | Value |
|-------|-------|
| Logical name | Problem A Severity Labels |
| Classification | PROCESSED |
| Source | ASRS reports with severity rubric applied |
| Produced by | `scripts/run_phase1.py` |
| Consumed by | `scripts/run_phase2.py`, `scripts/run_optuna_tuning.py`, `notebooks/01a_rubric_iteration.ipynb`, `notebooks/06a_problem_A_severity_viz.ipynb` |
| Expected shape | ~38,000 rows x 2 cols (acn_num_ACN, severity_level) |
| Git status | LOCAL ONLY |
| Regeneration | Run `scripts/run_phase1.py` |

---

## PROCESSED DATA — PROBLEM B

### `data/processed/problem_b/category_targets.parquet`
| Field | Value |
|-------|-------|
| Logical name | Problem B Multi-Label Category Targets |
| Classification | PROCESSED |
| Source | ASRS reports with category taxonomy applied |
| Produced by | `scripts/run_phase1.py` |
| Consumed by | `scripts/run_phase3.py`, `streamlit_app/utils/loaders.py`, `streamlit_app/pages/2_Problem_B_Category.py`, `notebooks/01b_taxonomy_definition.ipynb`, `notebooks/06b_problem_B_category_viz.ipynb` |
| Expected shape | ~38,000 rows x 7 cols (ACN, primary_category, 5 binary label cols) |
| Git status | LOCAL ONLY |
| Regeneration | Run `scripts/run_phase1.py` |

---

## PROCESSED DATA — PROBLEM C

### `data/processed/problem_c/preflight_features_final.parquet`
| Field | Value |
|-------|-------|
| Logical name | Problem C Pre-Flight Risk Features |
| Classification | PROCESSED |
| Source | Case-control dataset with engineered features |
| Produced by | `notebooks/05e_preflight_features.ipynb` |
| Consumed by | `scripts/run_phase6.py`, `scripts/run_phase7.py`, `scripts/benchmark_catboost.py`, `scripts/evaluate_preflight.py`, `scripts/generate_figures.py`, `scripts/rebuild_preflight_labels.py`, `streamlit_app/utils/loaders.py`, `notebooks/05f_preflight_model.ipynb`, `notebooks/06c_problem_C_preflight_viz.ipynb` |
| Expected shape | ~1,012,278 rows x 46 cols |
| Git status | LOCAL ONLY |
| Regeneration | Run `notebooks/05e_preflight_features.ipynb` (requires weather enriched data) |

---

## RESULT FILES (stay in `data/processed/` — not reorganized)

| File | Classification | Produced by | Consumed by | Git status |
|------|---------------|-------------|-------------|-----------|
| `data/processed/emerging_risks.csv` | RESULT | `scripts/run_phase4.py` | Streamlit D, Backend, `tests/test_sanity.py` | **TRACKED** |
| `data/processed/topic_trends.parquet` | RESULT | `scripts/run_phase4.py` | Streamlit D, Backend | LOCAL ONLY |
| `data/processed/topic_changepoints.json` | RESULT | `scripts/run_phase4.py` | Streamlit D, Backend, `scripts/generate_figures.py` | LOCAL ONLY |
| `data/processed/topic_representations.json` | RESULT | `scripts/run_phase4.py` | Streamlit D, Backend | LOCAL ONLY |
| `data/processed/factor_graph.json` | RESULT | `scripts/run_phase5.py` | Streamlit E, Backend, `scripts/generate_figures.py` | **TRACKED** |
| `data/processed/centrality_report.csv` | RESULT | `scripts/run_phase5.py` | Streamlit E, Backend, `src/models/graph_analysis.py` | LOCAL ONLY |
| `data/processed/factor_patterns.json` | RESULT | `scripts/run_phase5.py` | Streamlit E, Backend | LOCAL ONLY |
| `data/processed/optuna.db` | TUNING ARTIFACT | `scripts/run_optuna_tuning.py` | Reference only | **TRACKED** |
