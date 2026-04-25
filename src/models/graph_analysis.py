"""
Problem E — Multi-Layer Contributing Factor Graph
====================================================
Builds a heterogeneous co-occurrence graph with multiple node types:
  - Contributing Factors (from ASRS taxonomy)
  - Flight Phase (from Aircraft 1.9_Flight Phase)
  - Aircraft Type (from Aircraft 1.2_Make Model Name, bucketed)
  - FAR Part (from Aircraft 1.5_Operating Under FAR Part)

Edges represent co-occurrence within the same incident report.
Community detection (Louvain) identifies systemic risk clusters.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import Counter
import json


# ─────────────────────────────────────────────────────────────
# Node Extraction Helpers
# ─────────────────────────────────────────────────────────────

def _parse_factors(val: str) -> List[str]:
    """Parse semicolon-delimited contributing factors into a list."""
    if pd.isna(val) or not str(val).strip():
        return []
    return [f"FACTOR:{f.strip()}" for f in str(val).split(";") if f.strip() and f.strip() != "nan"]


def _parse_flight_phase(val: str) -> Optional[str]:
    """Bucket flight phase into coarse categories."""
    if pd.isna(val) or not str(val).strip():
        return None
    phase = str(val).strip().lower()
    # Map fine-grained phases to coarse buckets
    phase_map = {
        "takeoff": "PHASE:Takeoff", "initial climb": "PHASE:Climb",
        "climb": "PHASE:Climb", "cruise": "PHASE:Cruise",
        "descent": "PHASE:Descent", "initial approach": "PHASE:Approach",
        "final approach": "PHASE:Approach", "approach": "PHASE:Approach",
        "landing": "PHASE:Landing", "taxi": "PHASE:Taxi",
        "parked": "PHASE:Ground", "standing": "PHASE:Ground",
    }
    for key, bucket in phase_map.items():
        if key in phase:
            return bucket
    return f"PHASE:{phase.title()[:30]}"


def _parse_aircraft_type(val: str) -> Optional[str]:
    """Bucket aircraft make/model into families."""
    if pd.isna(val) or not str(val).strip():
        return None
    model = str(val).strip().upper()
    # Map common aircraft families
    families = [
        ("B737", "AIRCRAFT:B737"), ("B747", "AIRCRAFT:B747"),
        ("B757", "AIRCRAFT:B757"), ("B767", "AIRCRAFT:B767"),
        ("B777", "AIRCRAFT:B777"), ("B787", "AIRCRAFT:B787"),
        ("A318", "AIRCRAFT:A320_Family"), ("A319", "AIRCRAFT:A320_Family"),
        ("A320", "AIRCRAFT:A320_Family"), ("A321", "AIRCRAFT:A320_Family"),
        ("A330", "AIRCRAFT:A330"), ("A340", "AIRCRAFT:A340"),
        ("A350", "AIRCRAFT:A350"), ("A380", "AIRCRAFT:A380"),
        ("CRJ", "AIRCRAFT:CRJ"), ("ERJ", "AIRCRAFT:ERJ"),
        ("EMB", "AIRCRAFT:ERJ"), ("MD-", "AIRCRAFT:MD"),
        ("ATR", "AIRCRAFT:ATR"), ("DHC", "AIRCRAFT:DHC"),
        ("DASH", "AIRCRAFT:DHC"), ("C172", "AIRCRAFT:Cessna_SE"),
        ("C182", "AIRCRAFT:Cessna_SE"), ("C208", "AIRCRAFT:Cessna_Caravan"),
        ("C206", "AIRCRAFT:Cessna_SE"), ("PA-", "AIRCRAFT:Piper"),
        ("SR22", "AIRCRAFT:Cirrus"), ("SR20", "AIRCRAFT:Cirrus"),
        ("BELL", "AIRCRAFT:Helicopter"), ("SIKORSKY", "AIRCRAFT:Helicopter"),
        ("R22", "AIRCRAFT:Helicopter"), ("R44", "AIRCRAFT:Helicopter"),
        ("UAV", "AIRCRAFT:UAS"), ("UAS", "AIRCRAFT:UAS"),
    ]
    for pattern, family in families:
        if pattern in model:
            return family
    return "AIRCRAFT:Other"


def _parse_far_part(val: str) -> Optional[str]:
    """Extract FAR Part from operating rules."""
    if pd.isna(val) or not str(val).strip():
        return None
    part = str(val).strip()
    if part in ("Part 121", "Part 91", "Part 135", "Part 129", "Part 107"):
        return f"FAR:{part}"
    return None


# ─────────────────────────────────────────────────────────────
# Graph Construction
# ─────────────────────────────────────────────────────────────

def build_multilayer_graph(
    df: pd.DataFrame,
    factor_col: str = "Assessments_Contributing Factors / Situations",
    phase_col: str = "Aircraft 1.9_Flight Phase",
    aircraft_col: str = "Aircraft 1.2_Make Model Name",
    far_col: str = "Aircraft 1.5_Operating Under FAR Part",
    severity_col: str = "severity_level",
    min_edge_weight: int = 5,
) -> nx.Graph:
    """
    Builds a heterogeneous co-occurrence graph with typed nodes.

    Node types (encoded in prefix):
      - FACTOR:  Contributing factor from ASRS taxonomy
      - PHASE:   Coarse flight phase bucket
      - AIRCRAFT: Aircraft family bucket
      - FAR:     FAR Part operating rule

    Edge attributes:
      - weight: co-occurrence count
      - severity_sum / avg_severity: cumulative/mean severity
    """
    G = nx.Graph()

    for _, row in df.iterrows():
        severity = row.get(severity_col, 0)

        # Collect all typed nodes for this report
        nodes = []
        nodes.extend(_parse_factors(row.get(factor_col, "")))

        phase = _parse_flight_phase(row.get(phase_col, ""))
        if phase:
            nodes.append(phase)

        aircraft = _parse_aircraft_type(row.get(aircraft_col, ""))
        if aircraft:
            nodes.append(aircraft)

        far = _parse_far_part(row.get(far_col, ""))
        if far:
            nodes.append(far)

        # Skip reports with < 2 nodes (no edges possible)
        if len(nodes) < 2:
            continue

        # Update nodes
        for n in nodes:
            if not G.has_node(n):
                ntype = n.split(":")[0]
                G.add_node(n, count=0, severity_sum=0.0, node_type=ntype)
            G.nodes[n]["count"] += 1
            G.nodes[n]["severity_sum"] += severity

        # Update edges (all pairs within this report)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                n1, n2 = nodes[i], nodes[j]
                if G.has_edge(n1, n2):
                    G[n1][n2]["weight"] += 1
                    G[n1][n2]["severity_sum"] += severity
                else:
                    G.add_edge(n1, n2, weight=1, severity_sum=float(severity))

    # Prune low-weight edges
    to_remove = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < min_edge_weight]
    G.remove_edges_from(to_remove)

    # Remove isolated nodes after pruning
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    # Compute averages
    for n, d in G.nodes(data=True):
        if d["count"] > 0:
            d["avg_severity"] = d["severity_sum"] / d["count"]
    for u, v, d in G.edges(data=True):
        d["avg_severity"] = d["severity_sum"] / d["weight"]

    return G


def detect_communities(G: nx.Graph) -> Dict[str, int]:
    """
    Louvain community detection on the graph.
    Returns a dict mapping node → community_id.
    """
    try:
        from networkx.algorithms.community import louvain_communities
        communities = louvain_communities(G, weight="weight", seed=42)
        mapping = {}
        for cid, members in enumerate(communities):
            for node in members:
                mapping[node] = cid
        return mapping
    except Exception:
        return {n: 0 for n in G.nodes()}


def find_critical_paths(
    G: nx.Graph,
    min_severity: float = 1.5,
    top_k: int = 20,
) -> List[Dict]:
    """
    Finds edges with highest average severity (high-risk co-occurrence patterns).
    """
    critical = []
    for u, v, d in G.edges(data=True):
        if d.get("avg_severity", 0) >= min_severity:
            critical.append({
                "source": u, "target": v,
                "weight": d["weight"],
                "avg_severity": d["avg_severity"],
                "source_type": G.nodes[u].get("node_type", "?"),
                "target_type": G.nodes[v].get("node_type", "?"),
            })
    critical.sort(key=lambda x: (x["avg_severity"], x["weight"]), reverse=True)
    return critical[:top_k]


def compute_centrality_report(G: nx.Graph, top_k: int = 15) -> pd.DataFrame:
    """
    Compute degree, betweenness, and weighted degree centrality.
    Returns a DataFrame sorted by weighted degree.
    """
    degree_cent = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight="weight")
    weighted_deg = {n: sum(d["weight"] for _, _, d in G.edges(n, data=True))
                    for n in G.nodes()}

    rows = []
    for n in G.nodes():
        rows.append({
            "node": n,
            "type": G.nodes[n].get("node_type", "?"),
            "count": G.nodes[n].get("count", 0),
            "avg_severity": G.nodes[n].get("avg_severity", 0),
            "degree_centrality": degree_cent[n],
            "betweenness": betweenness[n],
            "weighted_degree": weighted_deg[n],
        })

    df = pd.DataFrame(rows).sort_values("weighted_degree", ascending=False)
    return df.head(top_k)


def graph_summary(G: nx.Graph, communities: Dict[str, int]) -> Dict[str, Any]:
    """Compute summary statistics for the graph."""
    type_counts = Counter(d.get("node_type", "?") for _, d in G.nodes(data=True))
    n_communities = len(set(communities.values())) if communities else 0

    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "node_type_counts": dict(type_counts),
        "num_communities": n_communities,
        "avg_clustering": nx.average_clustering(G, weight="weight"),
    }


def export_graph_to_json(G: nx.Graph, path: Path):
    """Export graph to node-link JSON for React/D3 visualization."""
    data = nx.node_link_data(G)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Graph exported to {path}")
