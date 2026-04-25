# OSFDA — Open-Source Flight Data Analysis

> **Predictive aviation risk analysis system built on NASA's ASRS incident database.**  
> Multi-model pipeline for severity classification, category routing, emerging risk discovery, and factor graph analysis.

---

## Overview

OSFDA processes **38,655 Aviation Safety Reporting System (ASRS)** incident reports to solve five interconnected problems:

| Problem | Task | Approach | Key Metric |
|---------|------|----------|------------|
| **A** | Incident Severity Classification | CatBoost multiclass (4 ordinal levels) | QWK = 0.24 |
| **B** | Category Classification | 3-tier: TF-IDF → SBERT → PyTorch Fusion MLP | Macro-F1 = 0.62 |
| **C** | Primary Problem Routing | Deterministic taxonomy mapping | — |
| **D** | Emerging Risk Discovery | BERTopic + PELT changepoint detection | 30 topics, 13 changepoints |
| **E** | Contributing Factor Graph | Multi-layer graph + Louvain communities | 60 nodes, 4 types |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Windows / Linux / macOS
- ~4GB disk space (models + embeddings)

### Installation

```bash
git clone https://github.com/yashnayan8795/OSFDA.git
cd OSFDA
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### Set Environment

```powershell
# PowerShell (Windows)
$env:PYTHONPATH = "E:\OSFDA"

# Bash (Linux/macOS)
export PYTHONPATH=$(pwd)
```

### Run the Full Pipeline

```bash
# 1. Validate the foundation
python -m pytest tests/ -v

# 2. Data foundation — loads ASRS, applies severity rubric, creates temporal split
python scripts/run_phase1.py

# 3. Severity models — trains LightGBM + CatBoost, selects best
python scripts/run_phase2.py

# 4. Category models — TF-IDF baseline → SBERT text tower → Fusion MLP
python scripts/run_phase3.py

# 5. Emerging risks — BERTopic clustering + changepoint detection
python scripts/run_phase4.py

# 6. Factor graph — multi-layer graph + community detection
python scripts/run_phase5.py
```

Or run everything at once:

```bash
python -m src.pipeline
```

---

## Project Structure

```
OSFDA/
├── configs/                       # YAML configuration files
│   ├── main_config.yaml           # Dataset paths, project metadata
│   ├── category_taxonomy_v1.yaml  # 8-category incident taxonomy
│   ├── cost_matrix.yaml           # Asymmetric misclassification costs
│   └── feature_whitelist.yaml     # Leakage-safe feature definitions
│
├── data/
│   ├── raw/                       # Source ASRS data (parquet)
│   └── processed/                 # Engineered targets, embeddings, outputs
│
├── models/                        # Serialized trained models
│   ├── severity_catboost.cbm      # Best severity model
│   ├── severity_lgbm.txt          # LightGBM baseline
│   ├── bertopic_model             # Topic model (BERTopic)
│   └── category_*.joblib          # Category classification models
│
├── scripts/                       # Phase execution scripts
│   ├── run_phase1.py              # Data foundation
│   ├── run_phase2.py              # Severity (Problem A)
│   ├── run_phase3.py              # Category (Problem B)
│   ├── run_phase4.py              # Emerging risks (Problem D)
│   ├── run_phase5.py              # Factor graph (Problem E)
│   └── run_optuna_tuning.py       # Hyperparameter optimization
│
├── src/                           # Core library modules
│   ├── pipeline.py                # End-to-end orchestrator
│   ├── data/
│   │   ├── loader.py              # ASRS data loading
│   │   ├── target_engineering.py  # Severity rubric v2.1, category taxonomy
│   │   └── leakage_audit.py       # Feature whitelist enforcement
│   ├── features/
│   │   ├── encoding.py            # Categorical encoding, experience bucketing
│   │   ├── temporal.py            # Temporal split, cyclical features
│   │   └── text.py                # Narrative preprocessing, SBERT encoding
│   ├── models/
│   │   ├── severity.py            # LightGBM + CatBoost + calibration
│   │   ├── category.py            # TF-IDF / SBERT / PyTorch Fusion MLP
│   │   ├── discovery.py           # BERTopic + PELT changepoints
│   │   └── graph_analysis.py      # Multi-layer graph + Louvain + centrality
│   ├── evaluation/
│   │   ├── ordinal_metrics.py     # QWK, bootstrap CI, confusion matrix
│   │   ├── calibration.py         # Expected Calibration Error (ECE)
│   │   └── multilabel_metrics.py  # Multi-label F1, hamming loss
│   └── utils/
│       └── config.py              # Config loading, path resolution, seeds
│
├── tests/                         # Test suite (86 tests)
│   ├── test_sanity.py             # Cross-cutting pipeline sanity checks
│   ├── test_target_engineering.py # Rubric v2.1 + taxonomy tests
│   ├── test_leakage_audit.py      # Leakage detection tests
│   ├── test_category_model.py     # Category model tests
│   └── test_temporal_split.py     # Temporal split integrity tests
│
├── notebooks/                     # Exploratory notebooks (legacy)
├── requirements.txt
└── README.md
```

---

## Pipeline Architecture

```
Phase 1: Data Foundation
  └─ ASRS Parquet → Severity Rubric v2.1 → Category Taxonomy → Leakage Audit → Temporal Split

Phase 2: Severity Classification (Problem A)
  └─ Feature Engineering → LightGBM + CatBoost → Isotonic Calibration → Model Comparison
     └─ Slice Analysis (by year) + Feature Importance

Phase 3: Category Classification (Problem B)
  └─ Tier 1: TF-IDF + Logistic Regression (baseline)
  └─ Tier 2: SBERT Embeddings + Logistic Regression (text tower)
  └─ Tier 3: SBERT + LightGBM Leaves → PyTorch Fusion MLP (BCEWithLogitsLoss)

Phase 4: Emerging Risk Discovery (Problem D)
  └─ SBERT Embeddings → PCA(15) → KMeans → BERTopic (30 topics)
     └─ Monthly Counts → PELT Changepoint Detection (penalty=3.0)
     └─ Risk Score = Growth × Severity × Changepoint Multiplier

Phase 5: Contributing Factor Graph (Problem E)
  └─ Multi-layer nodes: FACTOR, PHASE, AIRCRAFT, FAR Part
     └─ Louvain Community Detection → Centrality Analysis → Critical Paths
```

---

## Key Design Decisions

### Severity Rubric (v2.1)

The severity rubric is a **deterministic function** mapping post-incident outcome fields to ordinal levels 0–3. It does **not** use ML — this prevents circular reasoning (predicting severity from severity).

Key changes in v2.1:
- "Landed in emergency condition" demoted from Level 3 → Level 2
- "Malfunctioning" component removed entirely (too broad)
- Component failure requires *both* `Failed` + `Equipment Problem Critical` for Level 2

### Leakage Prevention

Post-incident fields (`Events.5_Result`, `Events_Anomaly`, `Component.3_Problem`) are used to **define** severity but are **never** included as features. A whitelist-based audit validates this at every run.

### Temporal Split

Train/Val/Test are split **by date** (not random), preventing future data from leaking into training:
- Train: 2012–2018
- Val: 2018–2019
- Test: 2019–2022

### UMAP Workaround

UMAP requires `numba`, which may be blocked by enterprise security policies (Windows Application Control). The pipeline degrades gracefully:
- **PCA(15)** replaces UMAP for dimensionality reduction
- **MiniBatchKMeans + hierarchical merging** replaces HDBSCAN for topic clustering
- **Built-in feature importance** replaces SHAP for explainability

---

## Model Results

### Problem A — Severity Classification

| Model | QWK | Macro-F1 | ECE (avg) |
|-------|-----|----------|-----------|
| LightGBM | 0.2245 | 0.3434 | 0.030 |
| **CatBoost** | **0.2387** | **0.3488** | **0.028** |

### Problem B — Category Classification

| Tier | Model | Macro-F1 |
|------|-------|----------|
| 1 | TF-IDF + LogReg | ~0.45 |
| 2 | SBERT + LogReg | ~0.52 |
| **3** | **Fusion MLP** | **0.6218** |

### Problem D — Top Emerging Risks

| Topic | Risk Score | Growth | Theme |
|-------|-----------|--------|-------|
| 21 | 4.79 | 3.12× | Tire/gear/wheel failures |
| 3 | 4.48 | 1.52× | Smoke/odor in cabin (changepoint detected) |
| 12 | 3.60 | 2.22× | Flap deployment issues |
| 6 | 2.20 | 1.95× | GPS jamming |

### Problem E — Top Critical Co-occurrences

| Pattern | Avg Severity | Count |
|---------|-------------|-------|
| Landing + Helicopter | 2.40 | 10 |
| Part 91 + Traffic Pattern | 1.90 | 10 |
| Approach + Software/Automation | 1.89 | 46 |

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_sanity.py -v          # 18 cross-cutting checks
python -m pytest tests/test_target_engineering.py  # Rubric + taxonomy
python -m pytest tests/test_leakage_audit.py       # Leakage validation
```

Current status: **86/86 tests passing**

---

## Data

The ASRS dataset is loaded from HuggingFace:

```python
from datasets import load_dataset
dataset = load_dataset("shayangerami/asrs")
```

It contains 38,655 incident reports with 90+ columns including narratives, metadata, anomaly classifications, and contributing factors.

---

## Known Limitations

1. **QWK for severity is modest (0.24)** — the model predicts from metadata columns only, without access to the post-incident outcome fields that define severity. This is by design (no leakage), but fundamentally limits accuracy.

2. **UMAP/SHAP unavailable** — blocked by Numba DLL policy on some Windows environments. Pipeline degrades gracefully.

3. **Problem C is implicit** — primary problem routing is a deterministic taxonomy lookup, not a trained classifier.

4. **No API/Frontend** — the system runs as batch scripts. FastAPI + React integration is planned but not implemented.

---

## License

This project is for educational and research purposes.

## Acknowledgments

- **NASA ASRS** — Aviation Safety Reporting System
- **NTSB** — Safety classification standards (49 CFR 830.2)
- **FAA** — Order 7210.56 (separation categories)
