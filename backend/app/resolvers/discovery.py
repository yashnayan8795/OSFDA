import strawberry
from typing import List
from app.types.discovery import EmergingRisk, TrendPoint, TopicDetail, ChangepointAlert, RiskSort
from app.services import discovery_svc

@strawberry.type
class DiscoveryQuery:
    @strawberry.field
    def emerging_risks(self, limit: int = 10, sort_by: RiskSort = RiskSort.RISK_SCORE) -> List[EmergingRisk]:
        data = discovery_svc.get_emerging_risks(limit, sort_by.value)
        return [EmergingRisk(**d) for d in data]

    @strawberry.field
    def risk_trend(self, topic_id: int) -> List[TrendPoint]:
        data = discovery_svc.get_risk_trend(topic_id)
        return [TrendPoint(**d) for d in data]

    @strawberry.field
    def topic_detail(self, topic_id: int) -> TopicDetail:
        data = discovery_svc.get_topic_detail(topic_id)
        return TopicDetail(**data)

    @strawberry.field
    def changepoint_alerts(self) -> List[ChangepointAlert]:
        data = discovery_svc.get_changepoint_alerts()
        return [ChangepointAlert(**d) for d in data]
