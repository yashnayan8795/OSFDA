"""
Phase 5 — Problem E: Contributing Factor Graph Analysis
=========================================================
Builds a semantic co-occurrence graph of contributing factors
and extracts high-severity critical paths.

Usage:
    $env:PYTHONPATH = "E:\OSFDA"
    python scripts/run_phase5.py
"""

import time
import networkx as nx
from pathlib import Path

from src.utils.config import load_main_config, resolve_path, set_seeds
from src.data.loader import load_raw_data
from src.data.target_engineering import apply_severity_rubric
from src.models.graph_analysis import build_factor_graph, find_critical_paths, export_graph_to_json

def banner(text):
    print(f"\n{'='*60}\n  {text}\n{'='*60}")

def main():
    set_seeds()
    banner("Phase 5 — Problem E: Contributing Factor Graph")

    print("Loading data...")
    config = load_main_config()
    df = load_raw_data(config)
    df = apply_severity_rubric(df)

    banner("1. Building Co-occurrence Graph")
    start = time.time()
    
    # We use Assessments_Contributing Factors / Situations
    G = build_factor_graph(df, min_edge_weight=10)
    
    print(f"  Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # Save Graph
    export_path = resolve_path("data/processed/factor_graph.json")
    export_graph_to_json(G, export_path)

    banner("2. Extracting Critical Paths (High Severity Patterns)")
    # Using 1.5 as minimum severity because 2 is quite high for an average over many reports
    critical_paths = find_critical_paths(G, min_severity=1.5, top_k=20)
    
    for i, path in enumerate(critical_paths, 1):
        print(f"  {i:>2}. {path['source']} <--> {path['target']}")
        print(f"      Severity: {path['avg_severity']:.2f} | Co-occurrences: {path['weight']}")

    elapsed = time.time() - start
    print(f"\n  Phase 5 done in {elapsed:.1f}s")

if __name__ == "__main__":
    main()
