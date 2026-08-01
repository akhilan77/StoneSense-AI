"""Service layer package for mock contract responses.

    The architecture keeps this layer isolated from API routing so that
    higher-level layers depend on typing and behavior contracts instead of
    implementation-specific details.
"""

from app.services.assessment_service import AssessmentService
from app.services.prediction_service import PredictionService

__all__ = ["AssessmentService", "PredictionService"]
