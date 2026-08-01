from app.schemas.patient import PatientInformation
from app.schemas.responses import ExplainabilityResponse, RiskPredictionResponse, StoneDetectionResponse


class PredictionService:
    """Mock prediction service for Day 2 API contract definition.

    The service intentionally returns deterministic mock payloads to
    decouple the API contract from future ML and DL implementations.
    """

    @staticmethod
    def get_mock_risk_prediction(payload: PatientInformation) -> RiskPredictionResponse:
        """Return a mocked patient risk response.

        Args:
            payload: Request payload for patient information.

        Returns:
            RiskPredictionResponse: Structured mock prediction.
        """
        return RiskPredictionResponse(
            probability=0.78,
            risk_level="high",
            confidence=0.91,
        )

    @staticmethod
    def get_mock_detection_response() -> StoneDetectionResponse:
        """Return a mocked image-based stone detection response."""
        return StoneDetectionResponse(
            detected=True,
            confidence=0.89,
            stone_size="5-7 mm",
            stone_location="left ureter",
        )

    @staticmethod
    def get_mock_explainability() -> ExplainabilityResponse:
        """Return a mocked explainability payload for the response contract."""
        return ExplainabilityResponse(
            shap_values={
                "age": 0.14,
                "bmi": 0.11,
                "blood_pressure": 0.18,
                "water_intake": -0.09,
            },
            lime_explanation="Feature importance indicates elevated blood pressure and low hydration are the dominant contributors.",
            gradcam_image_url="https://example.com/mock-gradcam.png",
        )
