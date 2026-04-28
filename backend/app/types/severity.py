import strawberry
from typing import List, Optional
from app.types.common import FeatureSpec, ModelInfo

@strawberry.input
class SeverityInput:
    # This will be dynamic in the future, but for the typed input we need some fields.
    # We can use a JSON string or a list of KeyValue pairs for full dynamism,
    # but for now we'll provide the common ones as optional.
    time_date: Optional[str] = None
    time_of_day: Optional[str] = None
    flight_conditions: Optional[str] = None
    weather_elements: Optional[str] = None
    light: Optional[str] = None
    ceiling: Optional[str] = None
    aircraft_make: Optional[str] = None
    aircraft_model: Optional[str] = None
    operator: Optional[str] = None
    far_part: Optional[str] = None
    flight_plan: Optional[str] = None
    mission: Optional[str] = None
    flight_phase: Optional[str] = None
    num_seats: Optional[int] = None
    crew_size: Optional[int] = None
    person_function: Optional[str] = None
    experience_hours: Optional[float] = None
    locale_reference: Optional[str] = None
    state_reference: Optional[str] = None

@strawberry.type
class SeverityResult:
    level: int
    label: str
    probabilities: List[float]
    confidence: float
    cost_sensitive_recommendation: str

@strawberry.type
class SeverityStats:
    minor: int
    moderate: int
    substantial: int
    critical: int
    total: int
