import strawberry
from typing import List
from app.types.graph import FactorGraph, GraphNode, FactorPattern, ModelMetrics
from app.services import graph_svc

@strawberry.type
class GraphQuery:
    @strawberry.field
    def factor_graph(self, min_weight: int = 5, max_nodes: int = 200) -> FactorGraph:
        data = graph_svc.get_factor_graph(min_weight, max_nodes)
        from app.types.graph import GraphEdge
        return FactorGraph(
            nodes=[GraphNode(**n) for n in data["nodes"]],
            edges=[GraphEdge(**e) for e in data["edges"]]
        )

    @strawberry.field
    def node_neighborhood(self, node_id: str, depth: int = 1) -> FactorGraph:
        data = graph_svc.get_node_neighborhood(node_id, depth)
        from app.types.graph import GraphEdge
        return FactorGraph(
            nodes=[GraphNode(**n) for n in data["nodes"]],
            edges=[GraphEdge(**e) for e in data["edges"]]
        )

    @strawberry.field
    def top_central_nodes(self, limit: int = 20) -> List[GraphNode]:
        data = graph_svc.get_top_central_nodes(limit)
        return [GraphNode(**n) for n in data]

    @strawberry.field
    def factor_patterns(self) -> List[FactorPattern]:
        data = graph_svc.get_factor_patterns()
        return [FactorPattern(**d) for d in data]

    @strawberry.field
    def model_metrics(self) -> ModelMetrics:
        data = graph_svc.get_model_metrics()
        return ModelMetrics(**data)
