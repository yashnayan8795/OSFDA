import strawberry
from app.resolvers.severity import SeverityQuery, SeverityMutation
from app.resolvers.category import CategoryQuery, CategoryMutation
from app.resolvers.preflight import PreflightQuery, PreflightMutation
from app.resolvers.discovery import DiscoveryQuery
from app.resolvers.graph import GraphQuery

@strawberry.type
class Query(
    SeverityQuery,
    CategoryQuery,
    PreflightQuery,
    DiscoveryQuery,
    GraphQuery
):
    pass

@strawberry.type
class Mutation(
    SeverityMutation,
    CategoryMutation,
    PreflightMutation
):
    pass

schema = strawberry.Schema(query=Query, mutation=Mutation)
