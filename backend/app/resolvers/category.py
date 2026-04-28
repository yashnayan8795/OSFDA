import strawberry
from typing import List
from app.types.category import CategoryInput, CategoryResult, CategoryCount, CategoryLabelSpec
from app.types.common import ModelInfo
from app.services import category_svc
from app.services.model_loader import ModelStore

@strawberry.type
class CategoryQuery:
    @strawberry.field
    def category_distribution(self) -> List[CategoryCount]:
        data = category_svc.get_category_distribution()
        return [CategoryCount(**d) for d in data]

    @strawberry.field
    def category_labels(self) -> List[CategoryLabelSpec]:
        data = category_svc.get_category_labels()
        return [CategoryLabelSpec(**d) for d in data]

    @strawberry.field
    def category_model_info(self) -> ModelInfo:
        info = ModelStore.category_info or {
            "status": "UNAVAILABLE", "version": "N/A", "trained_at": "N/A",
            "primary_metric_name": "Macro-F1", "primary_metric_value": 0.0, "calibrated": False
        }
        return ModelInfo(**info)

@strawberry.type
class CategoryMutation:
    @strawberry.mutation
    def classify_categories(self, input: CategoryInput) -> CategoryResult:
        result = category_svc.classify_categories(input.narrative, input.synopsis, input.accurate)
        return CategoryResult(**result)
