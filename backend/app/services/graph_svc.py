from app.services.model_loader import ModelStore

def get_factor_graph(min_weight: int = 5, max_nodes: int = 200) -> dict:
    nodes = ModelStore.graph_nodes or []
    edges = ModelStore.graph_edges or []
    
    # Filter edges by weight
    filtered_edges = [e for e in edges if e.get("weight", 0) >= min_weight]
    
    # Limit nodes to those connected by filtered edges, up to max_nodes
    connected_node_ids = set()
    for e in filtered_edges:
        connected_node_ids.add(e["source"])
        connected_node_ids.add(e["target"])
        
    # Sort nodes by count and take top max_nodes
    relevant_nodes = [n for n in nodes if n["id"] in connected_node_ids]
    relevant_nodes.sort(key=lambda x: x.get("count", 0), reverse=True)
    top_nodes = relevant_nodes[:max_nodes]
    top_node_ids = {n["id"] for n in top_nodes}
    
    # Final edge filter to only include edges between top_nodes
    final_edges = [e for e in filtered_edges if e["source"] in top_node_ids and e["target"] in top_node_ids]
    
    return {
        "nodes": top_nodes,
        "edges": final_edges
    }

def get_node_neighborhood(node_id: str, depth: int = 1) -> dict:
    # Simplified depth=1 neighbor search
    nodes = ModelStore.graph_nodes or []
    edges = ModelStore.graph_edges or []
    
    neighbor_ids = {node_id}
    neighborhood_edges = []
    
    # Find direct neighbors
    for e in edges:
        if e["source"] == node_id or e["target"] == node_id:
            neighbor_ids.add(e["source"])
            neighbor_ids.add(e["target"])
            neighborhood_edges.append(e)
            
    neighborhood_nodes = [n for n in nodes if n["id"] in neighbor_ids]
    
    return {
        "nodes": neighborhood_nodes,
        "edges": neighborhood_edges
    }

def get_factor_patterns() -> list[dict]:
    return ModelStore.factor_patterns or []

def get_top_central_nodes(limit: int = 20) -> list[dict]:
    df = ModelStore.centrality_df
    if df is None: return []
    
    subset = df.head(limit)
    results = []
    for _, row in subset.iterrows():
        results.append({
            "id": row["node"],
            "count": int(row["count"]),
            "avg_severity": float(row["avg_severity"]),
            "node_type": row["type"]
        })
    return results

def get_model_metrics() -> dict:
    # Combine info from all model info files
    sev_info = ModelStore.severity_info or {}
    cat_info = ModelStore.category_info or {}
    
    return {
        "severity_qwk": float(sev_info.get("primary_metric_value", 0.0)),
        "severity_model": sev_info.get("status", "UNKNOWN"),
        "category_macro_f1": float(cat_info.get("primary_metric_value", 0.0)),
        "category_model": cat_info.get("status", "UNKNOWN"),
        "total_incidents": 38655,
        "graph_nodes": len(ModelStore.graph_nodes or []),
        "graph_edges": len(ModelStore.graph_edges or []),
        "emerging_topics": len(ModelStore.emerging_risks_df) if ModelStore.emerging_risks_df is not None else 0
    }
