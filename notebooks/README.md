# Notebook Layer Documentation

## 1. Purpose of the Notebook Layer

The `notebooks/` directory contains exploratory, research-prototyping, diagnostic, and visualization workflows supporting the OSFDA (Open-Source Flight Data Analytics) project.

In this repository's architectural design:
- **Notebooks** serve as interactive research workbenches for data discovery, rubric iterations, baseline prototyping, ablation analysis, and diagnostic visualizations.
- **Python modules (`src/`)** host the production-grade, tested, reusable logic and algorithms.
- **Pipeline scripts (`scripts/`)** provide deterministic, repeatable execution of batch processes (data preparation, model training, benchmarking, and figure generation).
- **Disk artifacts** (`data/`, `models/`, `results/`) act as the decoupled interface between pipeline stages. Notebooks never import or invoke other notebooks directly (`%run`, `papermill`, etc. = 0).

---

## 2. Directory Structure

The 18 research notebooks are organized by research problem and shared utility domains:

```
notebooks/
├── README.md
│
├── shared/                                     # Shared data foundations & audits
│   ├── 00_data_acquisition.ipynb               # Raw NASA ASRS download & inventory
│   ├── 01c_leakage_audit.ipynb                 # Label & temporal leakage diagnostic
│   ├── 01d_temporal_split_eda.ipynb            # Temporal split EDA & distributions
│   └── 01e_redaction_impact.ipynb              # Impact of text redaction on signals
│
├── problem_a/                                  # Problem A: Incident Severity Classification
│   ├── 01a_rubric_iteration.ipynb              # Severity rubric v1/v2 rule definition
│   └── 06a_problem_A_severity_viz.ipynb        # Severity model evaluation & confusion matrix
│
├── problem_b/                                  # Problem B: Incident Category Classification
│   ├── 01b_taxonomy_definition.ipynb           # Category taxonomy mapping & statistics
│   ├── 02b_category_model.ipynb                # Multi-label category baseline & SBERT ablation
│   └── 06b_problem_B_category_viz.ipynb        # Category classification performance viz
│
├── problem_c/                                  # Problem C: Pre-Flight Operational Risk Prediction
│   ├── 05a_ntsb_acquisition.ipynb              # NTSB accident database extraction
│   ├── 05b_bts_flights.ipynb                   # BTS on-time flight data acquisition
│   ├── 05c_case_control.ipynb                  # Case-control pairing (NTSB cases vs BTS controls)
│   ├── 05d_weather_enrichment.ipynb            # NOAA weather feature enrichment
│   ├── 05e_preflight_features.ipynb            # Final feature engineering & scaling
│   ├── 05f_preflight_model.ipynb               # LightGBM pre-flight risk model & calibration
│   └── 06c_problem_C_preflight_viz.ipynb       # ROC/PR curves & risk calibration viz
│
├── problem_d/                                  # Problem D: Emerging Risk Detection
│   └── 06d_problem_D_emerging_risks_viz.ipynb  # Topic model trends & changepoint viz
│
└── problem_e/                                  # Problem E: Factor Graph Analysis
    └── 06e_problem_E_factor_graph_viz.ipynb    # Incident causal factor graph & network viz
```

---

## 3. Research Problem Overview

| Problem | Title | Research Scope & Methodology |
|---------|-------|------------------------------|
| **Shared** | Foundational Data & Audits | Cross-cutting ASRS data acquisition, temporal split verification, leakage audit, and narrative redaction impact. |
| **Problem A** | Incident Severity Classification | 4-tier ordinal severity classification (Minor, Moderate, Substantial, Critical) derived from operational impact rubrics. |
| **Problem B** | Incident Category Classification | Multi-label classification across 5 operational anomaly categories using TF-IDF and Sentence-BERT fusion architectures. |
| **Problem C** | Pre-Flight Operational Risk | Contextual pre-departure risk prediction using retrospective case-control design (NTSB incident cases matched with commercial BTS flights, enriched with NOAA METAR weather). |
| **Problem D** | Emerging Risk Detection | Unsupervised topic modeling and temporal changepoint detection over narrative texts to identify emerging aviation safety hazards. |
| **Problem E** | Factor Graph Analysis | Graph-theoretic representation of co-occurring causal factors, chain-of-events analysis, and node centrality across incident categories. |

---

## 4. Execution & Data Dependency Matrix

> [!NOTE]
> **Pipeline Independence & Modular Execution:** The OSFDA repository contains two decoupled primary data streams:
> 1. **ASRS Pipeline (Problems A, B, D, E):** Sourced from NASA ASRS textual reports.
> 2. **Pre-Flight Pipeline (Problem C):** Sourced independently from NTSB accident reports and BTS flight records.
>
> These data streams share no raw data files and can be executed independently. Furthermore, notebooks are not intended to be run in a single global linear sequence; each problem area provides modular, self-contained workflows for data exploration, model training, or diagnostic evaluation.

| Notebook | Problem | Role | Primary Inputs | Primary Outputs |
|:---|:---:|:---:|:---|:---|
| `shared/00_data_acquisition.ipynb` | Shared | Data Ingestion | HuggingFace ASRS dataset | `data/raw/asrs/asrs_full.parquet`, `data/metadata/column_inventory.csv` |
| `shared/01c_leakage_audit.ipynb` | Shared | Diagnostic Audit | `data/raw/asrs/asrs_full.parquet` | Leakage audit diagnostic report (console/in-notebook) |
| `shared/01d_temporal_split_eda.ipynb` | Shared | Diagnostic EDA | `data/processed/shared/temporal_splits.parquet` | Split distribution statistics & temporal charts |
| `shared/01e_redaction_impact.ipynb` | Shared | Diagnostic Study | `data/raw/asrs/asrs_full.parquet` | Redaction token impact analysis |
| `problem_a/01a_rubric_iteration.ipynb` | Problem A | Target Prototyping | `data/raw/asrs/asrs_full.parquet` | `data/processed/problem_a/severity_targets.parquet` |
| `problem_a/06a_problem_A_severity_viz.ipynb` | Problem A | Evaluation Viz | `data/processed/problem_a/severity_targets.parquet`, `models/` | Confusion matrices, error distributions, severity charts |
| `problem_b/01b_taxonomy_definition.ipynb` | Problem B | Target Prototyping | `data/raw/asrs/asrs_full.parquet` | `data/processed/problem_b/category_targets.parquet` |
| `problem_b/02b_category_model.ipynb` | Problem B | Model Prototyping | `data/processed/shared/embeddings/emb_*.npy`, `category_targets.parquet` | Tier 1/2/3 model ablation metrics & SBERT checkpoints |
| `problem_b/06b_problem_B_category_viz.ipynb` | Problem B | Evaluation Viz | `data/processed/problem_b/category_targets.parquet`, `models/` | Multilabel precision-recall, per-class F1 performance |
| `problem_c/05a_ntsb_acquisition.ipynb` | Problem C | Data Ingestion | NTSB CAROL / `data/raw/ntsb/source/avall.zip` | `data/raw/ntsb/ntsb_accidents.parquet` |
| `problem_c/05b_bts_flights.ipynb` | Problem C | Data Ingestion | BTS TranStats monthly archives | `data/raw/bts/monthly/`, `data/raw/bts/annual/bts_20??.parquet` |
| `problem_c/05c_case_control.ipynb` | Problem C | Dataset Assembly | `data/raw/ntsb/ntsb_accidents.parquet`, `data/raw/bts/annual/bts_20??.parquet` | `data/interim/problem_c/preflight_casecontrol.parquet` |
| `problem_c/05d_weather_enrichment.ipynb` | Problem C | Feature Enrichment | `data/interim/problem_c/preflight_casecontrol.parquet`, NOAA API | `data/interim/problem_c/preflight_weather_enriched.parquet` |
| `problem_c/05e_preflight_features.ipynb` | Problem C | Feature Engineering | `data/interim/problem_c/preflight_weather_enriched.parquet` | `data/processed/problem_c/preflight_features_final.parquet` |
| `problem_c/05f_preflight_model.ipynb` | Problem C | Model Training | `data/processed/problem_c/preflight_features_final.parquet` | `models/preflight_lgbm_calibrated.joblib`, evaluation metrics |
| `problem_c/06c_problem_C_preflight_viz.ipynb` | Problem C | Evaluation Viz | `models/preflight_lgbm_calibrated.joblib`, `preflight_features_final.parquet` | ROC curves, PR curves, calibration curve figures |
| `problem_d/06d_problem_D_emerging_risks_viz.ipynb` | Problem D | Evaluation Viz | `data/processed/emerging_risks.csv`, `topic_trends.parquet` | Temporal topic trends, changepoint timelines |
| `problem_e/06e_problem_E_factor_graph_viz.ipynb` | Problem E | Evaluation Viz | `data/processed/factor_graph.json`, `centrality_report.csv` | Factor network graphs, co-occurrence diagrams |

---

## 5. Functional Categorization

### A. Data-Generating Notebooks
- `shared/00_data_acquisition.ipynb`
- `problem_a/01a_rubric_iteration.ipynb`
- `problem_b/01b_taxonomy_definition.ipynb`
- `problem_c/05a_ntsb_acquisition.ipynb`
- `problem_c/05b_bts_flights.ipynb`
- `problem_c/05c_case_control.ipynb`
- `problem_c/05d_weather_enrichment.ipynb`
- `problem_c/05e_preflight_features.ipynb`

### B. Model-Training & Prototyping Notebooks
- `problem_b/02b_category_model.ipynb`
- `problem_c/05f_preflight_model.ipynb`

### C. Diagnostic, Evaluation, and Visualization Notebooks
- `shared/01c_leakage_audit.ipynb`
- `shared/01d_temporal_split_eda.ipynb`
- `shared/01e_redaction_impact.ipynb`
- `problem_a/06a_problem_A_severity_viz.ipynb`
- `problem_b/06b_problem_B_category_viz.ipynb`
- `problem_c/06c_problem_C_preflight_viz.ipynb`
- `problem_d/06d_problem_D_emerging_risks_viz.ipynb`
- `problem_e/06e_problem_E_factor_graph_viz.ipynb`

---

## 6. Root Discovery & Working Directory Convention

Every notebook starts with a standardized preamble that discovers the repository root by ascending the directory tree until `pyproject.toml` is found:

```python
import os
from pathlib import Path

# Find project root (contains pyproject.toml)
root = Path.cwd()
while not (root / "pyproject.toml").exists():
    root = root.parent
os.chdir(root)
print(f"Working directory: {root}")
```

Because of this preamble:
- Subfolder movement does not affect directory resolution: the search ascends two levels (e.g. `notebooks/problem_c/` $\to$ `notebooks/` $\to$ project root) and sets `os.chdir(root)`.
- All `from src...` imports and relative paths (e.g. `data/...`, `models/...`) resolve consistently across local runs, batch scripts, and Jupyter kernels.
