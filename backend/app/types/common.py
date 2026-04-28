import strawberry
from typing import List, Optional

@strawberry.type
class FeatureSpec:
    name: str
    display_name: str
    type: str              # "categorical" | "numeric" | "date"
    allowed_values: Optional[List[str]] = None
    required: bool = False
    description: Optional[str] = None

@strawberry.type
class ModelInfo:
    status: str            # TRAINED | STUB | UNAVAILABLE
    version: str           # git SHA or training run ID
    trained_at: str        # ISO date
    primary_metric_name: str
    primary_metric_value: float
    calibrated: bool
