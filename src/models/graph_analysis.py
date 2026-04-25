import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Any
from pathlib import Path
import json

def build_factor_graph(df: pd.DataFrame, factor_col: str = "Assessments_Contributing Factors / Situations", severity_col: str = "severity_level", min_edge_weight: int = 5) -> nx.Graph:
    """
    Builds a co-occurrence graph of contributing factors.
    """
    G = nx.Graph()
    
    # Iterate through rows and build co-occurrences
    for _, row in df.iterrows():
        factors_str = str(row.get(factor_col, ""))
        if factors_str == "nan" or not factors_str.strip():
            continue
            
        # ASRS often separates factors with semicolons
        factors = [f.strip() for f in factors_str.split(";")]
        factors = [f for f in factors if f and f != "nan"]
        
        severity = row.get(severity_col, 0)
        
        # Add nodes and edges
        for i, f1 in enumerate(factors):
            if not G.has_node(f1):
                G.add_node(f1, count=0, severity_sum=0)
            G.nodes[f1]['count'] += 1
            G.nodes[f1]['severity_sum'] += severity
            
            for j in range(i + 1, len(factors)):
                f2 = factors[j]
                if not G.has_node(f2):
                    G.add_node(f2, count=0, severity_sum=0)
                    
                if G.has_edge(f1, f2):
                    G[f1][f2]['weight'] += 1
                    G[f1][f2]['severity_sum'] += severity
                else:
                    G.add_edge(f1, f2, weight=1, severity_sum=severity)
                    
    # Prune low-weight edges and compute averages
    edges_to_remove = []
    for u, v, data in list(G.edges(data=True)):
        if data['weight'] < min_edge_weight:
            edges_to_remove.append((u, v))
        else:
            data['avg_severity'] = data['severity_sum'] / data['weight']
            
    G.remove_edges_from(edges_to_remove)
    
    # Compute node averages
    for node, data in G.nodes(data=True):
        if data['count'] > 0:
            data['avg_severity'] = data['severity_sum'] / data['count']
            
    return G

def find_critical_paths(G: nx.Graph, min_severity: float = 2.0, top_k: int = 10) -> List[Dict]:
    """
    Finds the edges (patterns) that have the highest average severity.
    """
    critical_edges = []
    for u, v, data in G.edges(data=True):
        if data['avg_severity'] >= min_severity:
            critical_edges.append({
                "source": u,
                "target": v,
                "weight": data['weight'],
                "avg_severity": data['avg_severity']
            })
            
    # Sort by severity then weight
    critical_edges.sort(key=lambda x: (x['avg_severity'], x['weight']), reverse=True)
    return critical_edges[:top_k]

def export_graph_to_json(G: nx.Graph, path: Path):
    """
    Exports a NetworkX graph to a node-link JSON format suitable for React/D3.
    """
    data = nx.node_link_data(G)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Graph exported to {path}")
