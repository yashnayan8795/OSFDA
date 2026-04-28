import strawberry
from typing import List, Optional
from app.types.common import FeatureSpec, ModelInfo

@strawberry.input
class FlightInput:
    carrier: str
    origin: str
    destination: str
    flight_date: str
    dep_time: str
    aircraft_type: Optional[str] = None
    weather_forecast: Optional[str] = None

@strawberry.type
class Contributor:
    feature: str
    impact: float # SHAP value or contribution
    description: str

@strawberry.type
class KeyValue:
    key: str
    value: float

@strawberry.type
class PreflightResult:
    risk_score: float                # calibrated probability
    risk_tier: str                   # Low / Medium / High
    base_rate_multiple: float        # vs population avg
    top_contributors: List[Contributor]
    model_status: str                # TRAINED | STUB
    disclaimer: str

@strawberry.type
class PreflightBaseRates:
    overall: float
    by_carrier: List[KeyValue]
    by_origin: List[KeyValue]
