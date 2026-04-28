import strawberry
from typing import List
from app.types.severity import SeverityInput, SeverityResult, SeverityStats
from app.types.common import FeatureSpec, ModelInfo
from app.services import severity_svc
from app.services.model_loader import ModelStore

@strawberry.type
class SeverityQuery:
    @strawberry.field
    def severity_distribution(self) -> SeverityStats:
        data = severity_svc.get_severity_distribution()
        return SeverityStats(**data)

    @strawberry.field
    def severity_feature_schema(self) -> List[FeatureSpec]:
        data = severity_svc.get_severity_feature_schema()
        return [FeatureSpec(**d) for d in data]

    @strawberry.field
    def severity_model_info(self) -> ModelInfo:
        info = ModelStore.severity_info or {
            "status": "UNAVAILABLE", "version": "N/A", "trained_at": "N/A",
            "primary_metric_name": "QWK", "primary_metric_value": 0.0, "calibrated": False
        }
        return ModelInfo(**info)

@strawberry.type
class SeverityMutation:
    @strawberry.mutation
    def classify_severity(self, input: SeverityInput) -> SeverityResult:
        result = severity_svc.classify_severity(vars(input))
        return SeverityResult(**result)
