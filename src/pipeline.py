"""
Flight Risk Analysis — End-to-End Pipeline Runner
===================================================
Orchestrates: Data Acquisition → Target Engineering → Leakage Audit →
Temporal Split → Feature Engineering → Model Training → Evaluation.

Run from project root:
    python -m src.pipeline
"""

import sys
import time
import subprocess

from src.utils.config import (
    load_main_config, load_severity_rubric, load_category_taxonomy,
    load_feature_whitelist, load_cost_matrix, set_seeds, PROJECT_ROOT,
)


def banner(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def run_phase0():
    """Phase 0: Verify scaffolding and configuration."""
    banner("PHASE 0 — Scaffolding Verification")
    config = load_main_config()
    print(f"  Project: {config['project']['name']} v{config['project']['version']}")
    print(f"  Root: {PROJECT_ROOT}")

    rubric = load_severity_rubric()
    print(f"  Severity rubric: v{rubric['version']} ({len(rubric['levels'])} levels)")

    taxonomy = load_category_taxonomy()
    print(f"  Category taxonomy: v{taxonomy['version']} ({len(taxonomy['categories'])} categories)")

    whitelist = load_feature_whitelist()
    print(f"  Feature whitelist: v{whitelist['version']}")
    print(f"    Problem A: {len(whitelist['problem_a_whitelist'])} features")
    print(f"    Problem B extras: {len(whitelist['problem_b_extra'])} features")

    cost = load_cost_matrix()
    print(f"  Cost matrix: v{cost['version']}")

    print("\n  ✅ Phase 0 passed: All configs load successfully.")
    return config


def run_phase1():
    """Phase 1: Data Foundation — acquisition, targets, leakage audit, split."""
    banner("PHASE 1 — Data Foundation")
    print("  Running scripts/run_phase1.py ...")
    cmd = [sys.executable, "scripts/run_phase1.py"]
    subprocess.run(cmd, check=True)


def run_phase2():
    """Phase 2: Problem A — Severity Model Training & Evaluation."""
    banner("PHASE 2 — Problem A: Severity Classification")
    print("  Running scripts/run_phase2.py ...")
    cmd = [sys.executable, "scripts/run_phase2.py"]
    subprocess.run(cmd, check=True)


def run_phase3():
    """Phase 3: Problem B — Category Classification."""
    banner("PHASE 3 — Problem B: Category Classification")
    print("  Running scripts/run_phase3.py (TF-IDF Baseline & SBERT Text Tower)...")
    cmd = [sys.executable, "scripts/run_phase3.py", "--tier", "all"]
    subprocess.run(cmd, check=True)


def run_phase4():
    """Phase 4: Problem D — Emerging Risk Discovery."""
    banner("PHASE 4 — Problem D: Emerging Risk Discovery")
    print("  Running scripts/run_phase4.py (BERTopic & Changepoint Detection)...")
    cmd = [sys.executable, "scripts/run_phase4.py"]
    subprocess.run(cmd, check=True)


def run_phase5():
    """Phase 5: Problem E — Contributing Factor Graph."""
    banner("PHASE 5 — Problem E: Contributing Factor Graph")
    print("  Running scripts/run_phase5.py (Semantic Graph & Critical Paths)...")
    cmd = [sys.executable, "scripts/run_phase5.py"]
    subprocess.run(cmd, check=True)


def main():
    """Run the full pipeline."""
    start = time.time()
    banner("Flight Risk Analysis — Full Pipeline")
    set_seeds()

    config = run_phase0()
    run_phase1()
    run_phase2()
    run_phase3()
    run_phase4()
    run_phase5()

    elapsed = time.time() - start
    banner(f"PIPELINE COMPLETE — {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
