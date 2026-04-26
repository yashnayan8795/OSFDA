"""
Phase 5 — Problem E: Multi-Layer Contributing Factor Graph
============================================================
Builds a heterogeneous co-occurrence graph, detects communities,
computes centrality, and extracts high-severity critical paths.

Usage:
    python scripts/run_phase5.py
"""

import time
import networkx as nx
import pandas as pd
from pathlib import Path

from src.utils.config import load_main_config, resolve_path, set_seeds
from src.data.loader import load_raw_data
from src.data.target_engineering import apply_severity_rubric
from src.models.graph_analysis import (
    build_multilayer_graph, detect_communities,
    find_critical_paths, compute_centrality_report,
    graph_summary, export_graph_to_json,
)


def banner(text):
    print(f"\n{'='*60}\n  {text}\n{'='*60}")


def main():
    set_seeds()
    banner("Phase 5 — Problem E: Multi-Layer Factor Graph")

    print("Loading data...")
    config = load_main_config()
    df = load_raw_data(config)
    df = apply_severity_rubric(df)

    # ──────────────────────────────────────────────────────────
    # 1. Build Multi-Layer Graph
    # ──────────────────────────────────────────────────────────
    banner("1. Building Multi-Layer Co-occurrence Graph")
    start = time.time()

    G = build_multilayer_graph(df, min_edge_weight=10)

    summary = graph_summary(G, {})
    print(f"  Nodes: {summary['num_nodes']}")
    print(f"  Edges: {summary['num_edges']}")
    print(f"  Density: {summary['density']:.4f}")
    print(f"  Node types: {summary['node_type_counts']}")

    # ──────────────────────────────────────────────────────────
    # 2. Community Detection (Louvain)
    # ──────────────────────────────────────────────────────────
    banner("2. Community Detection (Louvain)")

    communities = detect_communities(G)
    # Assign to nodes
    for node, cid in communities.items():
        G.nodes[node]["community"] = cid

    n_comm = len(set(communities.values()))
    print(f"  Detected {n_comm} communities.")

    # Print community composition
    comm_df = pd.DataFrame([
        {"node": n, "type": G.nodes[n].get("node_type"), "community": cid}
        for n, cid in communities.items()
    ])
    if not comm_df.empty:
        for cid in sorted(comm_df["community"].unique())[:5]:
            members = comm_df[comm_df["community"] == cid]
            type_dist = members["type"].value_counts().to_dict()
            sample = members["node"].head(5).tolist()
            print(f"    Community {cid}: {len(members)} nodes | Types: {type_dist}")
            print(f"      Sample: {', '.join(sample)}")

    # ──────────────────────────────────────────────────────────
    # 3. Centrality Analysis
    # ──────────────────────────────────────────────────────────
    banner("3. Centrality Analysis (Top 15)")

    centrality_df = compute_centrality_report(G, top_k=15)
    for _, row in centrality_df.iterrows():
        print(
            f"    {row['node']:<45} | Type={row['type']:<10} "
            f"| Sev={row['avg_severity']:.2f} | WtDeg={row['weighted_degree']:<6} "
            f"| Between={row['betweenness']:.4f}"
        )

    # ──────────────────────────────────────────────────────────
    # 4. Critical Paths (High-Severity Co-occurrences)
    # ──────────────────────────────────────────────────────────
    banner("4. Critical Paths (High-Severity Patterns)")

    critical = find_critical_paths(G, min_severity=1.5, top_k=20)
    if not critical:
        print("  No critical paths found (min_severity=1.5). Lowering to 1.0...")
        critical = find_critical_paths(G, min_severity=1.0, top_k=20)

    for i, p in enumerate(critical, 1):
        print(
            f"  {i:>2}. {p['source']:<35} <-> {p['target']:<35}"
        )
        print(
            f"      Severity: {p['avg_severity']:.2f} | "
            f"Co-occurrences: {p['weight']} | "
            f"Types: {p['source_type']}-{p['target_type']}"
        )

    # ──────────────────────────────────────────────────────────
    # 5. Export
    # ──────────────────────────────────────────────────────────
    banner("5. Export")

    export_path = resolve_path("data/processed/factor_graph.json")
    export_graph_to_json(G, export_path)

    # Save centrality report
    cent_path = resolve_path("data/processed/centrality_report.csv")
    centrality_df_full = compute_centrality_report(G, top_k=G.number_of_nodes())
    centrality_df_full.to_csv(cent_path, index=False)
    print(f"  Centrality report saved to {cent_path}")

    elapsed = time.time() - start
    print(f"\n  Phase 5 done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
