"""
Aviation Safety Analytics — Streamlit Demo
==========================================
Landing page. Use the sidebar to navigate between the five ML problems.
Run from project root:
    streamlit run streamlit_app/app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="Aviation Safety Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"

# ── Model / artefact status ──────────────────────────────────────────────────

def _check(path: Path) -> str:
    return "✅" if path.exists() else "❌"


with st.sidebar:
    st.markdown("## ✈️ Aviation Safety")
    st.markdown("### Artefact Status")
    st.markdown(
        f"""
| Problem | Status |
|---------|--------|
| A — Severity model | {_check(MODELS / 'severity_lgbm.txt')} |
| A — Calibrators | {_check(MODELS / 'severity_calibrators.joblib')} |
| B — TF-IDF model | {_check(MODELS / 'category_tfidf_baseline.joblib')} |
| B — SBERT tower | {_check(MODELS / 'category_text_tower.joblib')} |
| C — Preflight model | {_check(MODELS / 'preflight_lgbm_calibrated.joblib')} |
| D — Emerging risks | {_check(PROCESSED / 'emerging_risks.csv')} |
| D — Topic trends | {_check(PROCESSED / 'topic_trends.parquet')} |
| E — Factor graph | {_check(PROCESSED / 'factor_graph.json')} |
| E — Factor patterns | {_check(PROCESSED / 'factor_patterns.json')} |
        """
    )
    st.divider()
    st.caption(
        "Run the phase scripts to generate missing artefacts:\n\n"
        "`python scripts/run_phase2.py` — A\n\n"
        "`python scripts/run_phase3.py` — B + C\n\n"
        "`python scripts/run_phase4.py` — D\n\n"
        "`python scripts/run_phase5.py` — E"
    )

# ── Hero section ─────────────────────────────────────────────────────────────

st.title("✈️ Aviation Safety Analytics System")
st.markdown(
    "A multi-task ML system built on the **NASA ASRS corpus** (38,655 voluntary aviation "
    "incident reports, 2003–2020). Five interconnected problems covering severity triage, "
    "incident categorization, pre-flight risk scoring, emerging pattern discovery, and "
    "contributing factor graph mining."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("ASRS Reports", "38,655", "2003 – 2020")
c2.metric("Temporal Split", "Train / Val / Test", "2003-17 / 18 / 19-20")
c3.metric("Leakage Audit", "0 post-incident leaks", "strict whitelist")
c4.metric("Calibration", "Isotonic per-class", "QWK primary metric")

st.divider()

# ── Problem cards ─────────────────────────────────────────────────────────────

PROBLEMS = [
    (
        "🔴", "A — Severity Triage",
        "Ordinal severity **0–3** from physical outcome fields. "
        "LightGBM vs CatBoost with isotonic calibration and **cost-sensitive prediction**. "
        "Strict leakage audit: no post-incident features.",
        "Live prediction · Confusion matrix · What-If sliders",
        MODELS / "severity_lgbm.txt",
    ),
    (
        "🟡", "B — Incident Category",
        "Multi-label taxonomy (5 categories). "
        "Three-tier: TF-IDF baseline → Sentence-BERT text tower → Fusion MLP. "
        "Per-label threshold tuning on validation set.",
        "Narrative input → category chips · Tier comparison · Co-occurrence heatmap",
        MODELS / "category_tfidf_baseline.joblib",
    ),
    (
        "🟠", "C — Pre-Flight Risk",
        "Binary risk score from **BTS flights + NOAA weather + NTSB** join. "
        "Case-control sampling handles 5 % positive rate. "
        "Isotonic calibration.",
        "Flight params → risk gauge · ROC / PR curve · Sensitivity sliders",
        MODELS / "preflight_lgbm_calibrated.joblib",
    ),
    (
        "🟣", "D — Emerging Risks",
        "Unsupervised narrative clustering (BERTopic / PCA + MiniBatchKMeans) "
        "+ PELT changepoint detection. "
        "Severity-weighted risk scores surface rising themes before regulatory response.",
        "Risk ranking · Trend overlay · Changepoint markers · Topic browser",
        PROCESSED / "emerging_risks.csv",
    ),
    (
        "🔵", "E — Factor Graph",
        "Co-occurrence knowledge graph of **contributing factors, flight phases, "
        "aircraft types, and FAR parts**. "
        "Louvain community detection + betweenness centrality.",
        "Interactive network · Centrality table · High-severity edge patterns",
        PROCESSED / "factor_graph.json",
    ),
]

cols = st.columns(2)
for i, (emoji, title, description, features, artefact) in enumerate(PROBLEMS):
    ready = artefact.exists()
    badge = "✅ Ready" if ready else "❌ Run pipeline first"
    badge_color = "#2ecc71" if ready else "#e74c3c"
    with cols[i % 2]:
        with st.container(border=True):
            st.markdown(
                f"### {emoji} {title} "
                f'<span style="background:{badge_color};color:white;'
                f'padding:2px 8px;border-radius:8px;font-size:0.75em;">{badge}</span>',
                unsafe_allow_html=True,
            )
            st.caption(description)
            st.markdown(f"**Demo includes:** {features}")

st.divider()

# ── Architecture overview ─────────────────────────────────────────────────────

with st.expander("📐 System Architecture", expanded=False):
    st.markdown(
        """
        #### Data Flow
        ```
        ASRS (38k reports)  ──► Problem A: Severity labels ──► Problem D: severity-weighted trends
                            ──► Problem B: Category labels  ──► Problem E: factor-outcome links
                            ──► Problem D: Narrative embeds → BERTopic clusters → PELT changepoints
                            ──► Problem E: Contributing factors → co-occurrence graph

        BTS + NOAA + NTSB   ──► Problem C: Pre-flight risk  ◄── D trend features, A severity patterns
        ```

        #### Key Design Decisions
        | Decision | Choice | Rationale |
        |---|---|---|
        | Temporal split | Train 2003-17 / Val 2018 / Test 2019-20 | Prevent future-data leakage |
        | Primary metric | QWK (quadratic weighted kappa) | Penalises large ordinal jumps |
        | Severity prediction | Cost-sensitive (argmin expected cost) | Missing Level-3 costs 100× more |
        | Calibration | Isotonic regression per-class | Deployment-ready probabilities |
        | Topic modelling | PCA + MiniBatchKMeans (not UMAP) | Windows Numba DLL workaround |
        | Leakage gate | Feature whitelist YAML + audit script | All `Events.*` / `Assessments.*` excluded from Problem A |
        """
    )

st.caption("Navigate using the sidebar → select a problem page.")
