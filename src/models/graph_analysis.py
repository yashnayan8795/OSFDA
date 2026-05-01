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
import networkx as nx
from typing import Dict, List, Any, Optional
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
    Builds a heterogeneous co-occurrence graph with four node types:
      FACTOR, PHASE, AIRCRAFT, FAR.
    Edges connect any two entities that appear in the same incident report.
    Uses vectorized explode + merge for performance.
    """
    G = nx.Graph()

    # Assign a stable integer row_id across the whole DataFrame
    df = df.reset_index(drop=True)
    df["_row_id"] = df.index

    sev = df[["_row_id", severity_col]].rename(columns={severity_col: "_severity"})

    entity_parts: List[pd.DataFrame] = []

    # ── FACTOR nodes (semicolon-delimited list) ──────────────────
    if factor_col in df.columns:
        fwork = df[["_row_id", factor_col]].copy()
        fwork[factor_col] = fwork[factor_col].fillna("")
        fwork = fwork[fwork[factor_col].str.strip().astype(bool)]
        fwork["_factor_list"] = fwork[factor_col].str.split(";")
        exploded = fwork.explode("_factor_list")
        exploded["_factor_list"] = exploded["_factor_list"].str.strip()
        exploded = exploded[exploded["_factor_list"].astype(bool)]
        exploded["_entity"] = exploded["_factor_list"].apply(
            lambda v: _parse_factors(v)[0] if _parse_factors(v) else None
        )
        exploded = exploded.dropna(subset=["_entity"])
        exploded["_node_type"] = "FACTOR"
        entity_parts.append(exploded[["_row_id", "_entity", "_node_type"]])

    # ── PHASE nodes (single value per report) ────────────────────
    if phase_col in df.columns:
        pwork = df[["_row_id", phase_col]].copy()
        pwork["_entity"] = pwork[phase_col].apply(_parse_flight_phase)
        pwork = pwork.dropna(subset=["_entity"])
        pwork["_node_type"] = "PHASE"
        entity_parts.append(pwork[["_row_id", "_entity", "_node_type"]])

    # ── AIRCRAFT nodes (single value per report) ─────────────────
    if aircraft_col in df.columns:
        awork = df[["_row_id", aircraft_col]].copy()
        awork["_entity"] = awork[aircraft_col].apply(_parse_aircraft_type)
        awork = awork.dropna(subset=["_entity"])
        awork["_node_type"] = "AIRCRAFT"
        entity_parts.append(awork[["_row_id", "_entity", "_node_type"]])

    # ── FAR nodes (single value per report) ──────────────────────
    if far_col in df.columns:
        rwork = df[["_row_id", far_col]].copy()
        rwork["_entity"] = rwork[far_col].apply(_parse_far_part)
        rwork = rwork.dropna(subset=["_entity"])
        rwork["_node_type"] = "FAR"
        entity_parts.append(rwork[["_row_id", "_entity", "_node_type"]])

    if not entity_parts:
        return G

    all_entities = pd.concat(entity_parts, ignore_index=True)
    all_entities = all_entities.merge(sev, on="_row_id", how="left")
    all_entities["_severity"] = pd.to_numeric(all_entities["_severity"], errors="coerce").fillna(0)

    # ── Node statistics (count + avg_severity + node_type) ───────
    node_stats = (
        all_entities.groupby(["_entity", "_node_type"])
        .agg(count=("_row_id", "size"), severity_sum=("_severity", "sum"))
        .reset_index()
    )
    for _, nrow in node_stats.iterrows():
        G.add_node(
            nrow["_entity"],
            count=int(nrow["count"]),
            severity_sum=float(nrow["severity_sum"]),
            avg_severity=float(nrow["severity_sum"] / nrow["count"]),
            node_type=nrow["_node_type"],
        )

    # ── Edge statistics via self-join on _row_id ─────────────────
    pair_df = all_entities[["_row_id", "_entity", "_severity"]].copy()
    merged = pair_df.merge(pair_df, on="_row_id", suffixes=("_a", "_b"))
    merged = merged[merged["_entity_a"] < merged["_entity_b"]]

    edge_stats = merged.groupby(["_entity_a", "_entity_b"]).agg(
        weight=("_row_id", "size"),
        severity_sum=("_severity_a", "sum"),
    )
    edge_stats = edge_stats[edge_stats["weight"] >= min_edge_weight]
    edge_stats["avg_severity"] = edge_stats["severity_sum"] / edge_stats["weight"]

    for (f1, f2), erow in edge_stats.iterrows():
        G.add_edge(
            f1, f2,
            weight=int(erow["weight"]),
            severity_sum=float(erow["severity_sum"]),
            avg_severity=float(erow["avg_severity"]),
        )

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
