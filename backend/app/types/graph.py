import strawberry
from typing import List, Optional

@strawberry.type
class GraphNode:
    id: str
    count: int
    avg_severity: float
    node_type: Optional[str] = None
    community: Optional[int] = None

@strawberry.type
class GraphEdge:
    source: str
    target: str
    weight: int
    avg_severity: float

@strawberry.type
class FactorGraph:
    nodes: List[GraphNode]
    edges: List[GraphEdge]

@strawberry.type
class FactorPattern:
    pattern_id: str
    factors: List[str]
    support: int
    avg_severity: float
    lift: float

@strawberry.type
class ModelMetrics:
    severity_qwk: float
    severity_model: str
    category_macro_f1: float
    category_model: str
    total_incidents: int
    graph_nodes: int
    graph_edges: int
    emerging_topics: int
