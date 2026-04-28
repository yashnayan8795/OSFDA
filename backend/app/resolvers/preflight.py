import strawberry
from typing import List
from app.types.preflight import FlightInput, PreflightResult, PreflightBaseRates, Contributor
from app.types.common import FeatureSpec, ModelInfo
from app.services import preflight_svc
from app.services.model_loader import ModelStore

@strawberry.type
class PreflightQuery:
    @strawberry.field
    def preflight_feature_schema(self) -> List[FeatureSpec]:
        data = preflight_svc.get_preflight_feature_schema()
        return [FeatureSpec(**d) for d in data]

    @strawberry.field
    def preflight_model_info(self) -> ModelInfo:
        info = ModelStore.preflight_info or {
            "status": "UNAVAILABLE", "version": "N/A", "trained_at": "N/A",
            "primary_metric_name": "PR-AUC", "primary_metric_value": 0.0, "calibrated": False
        }
        return ModelInfo(**info)

    @strawberry.field
    def preflight_base_rates(self) -> PreflightBaseRates:
        data = preflight_svc.get_preflight_base_rates()
        # Handle empty lists for KV pairs
        return PreflightBaseRates(overall=data["overall"], by_carrier=[], by_origin=[])

@strawberry.type
class PreflightMutation:
    @strawberry.mutation
    def predict_preflight_risk(self, input: FlightInput) -> PreflightResult:
        result = preflight_svc.predict_preflight_risk(vars(input))
        # Ensure contributors are typed
        result["top_contributors"] = [Contributor(**c) for c in result["top_contributors"]]
        return PreflightResult(**result)
