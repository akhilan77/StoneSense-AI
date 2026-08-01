from app.schemas.patient import PatientInformation
from app.schemas.responses import FinalAssessmentResponse
from app.services.prediction_service import PredictionService


class AssessmentService:
    """Composite orchestration service for the final assessment response.

    This layer composes domain-specific mock responses into a single
    final assessment object, preserving a clean separation of concerns.
    """

    @staticmethod
    def build_assessment(patient: PatientInformation) -> FinalAssessmentResponse:
        """Build the final assessment object from the current mock contract.

        Args:
            patient: Patient information request object.

        Returns:
            FinalAssessmentResponse: Complete assessment payload.
        """
        risk_prediction = PredictionService.get_mock_risk_prediction(patient)
        stone_detection = PredictionService.get_mock_detection_response()
        explainability = PredictionService.get_mock_explainability()

        return FinalAssessmentResponse(
            patient=patient,
            risk_prediction=risk_prediction,
            stone_detection=stone_detection,
            explainability=explainability,
            recommendation=(
                "Increase water intake, monitor blood pressure, and schedule "
                "a follow-up clinical evaluation for stone analysis."
            ),
        )
