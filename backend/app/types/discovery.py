import strawberry
from typing import List, Optional
from enum import Enum

@strawberry.enum
class RiskSort(Enum):
    RISK_SCORE = "risk_score"
    GROWTH = "growth"
    SEVERITY = "severity"
    COUNT = "count"

@strawberry.type
class EmergingRisk:
    topic_id: int
    name: str
    risk_score: float
    growth_ratio: float
    recent_changepoint: bool
    avg_severity: float
    count: int

@strawberry.type
class TrendPoint:
    period: str
    count: int
    avg_severity: float

@strawberry.type
class TopicDetail:
    topic_id: int
    name: str
    keywords: List[str]
    sample_reports: List[str]
    description: Optional[str] = None

@strawberry.type
class ChangepointAlert:
    topic_id: int
    name: str
    date: str
    severity_association: float
    direction: str # UP | DOWN
