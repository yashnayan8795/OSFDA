import strawberry
from typing import List, Optional
from app.types.common import ModelInfo

@strawberry.input
class CategoryInput:
    narrative: str
    synopsis: Optional[str] = ""
    accurate: bool = False # Toggle for Fusion vs TF-IDF

@strawberry.type
class CategoryPrediction:
    label: str
    display_name: str
    probability: float
    predicted: bool
    threshold_used: float

@strawberry.type
class CategoryResult:
    predictions: List[CategoryPrediction]

@strawberry.type
class CategoryCount:
    label: str
    count: int

@strawberry.type
class CategoryLabelSpec:
    label: str
    display_name: str
    threshold: float
