from fastapi import FastAPI

from app.api.v1.routes.assessment_router import router as assessment_router
from app.api.v1.routes.health_router import router as health_router
from app.api.v1.routes.predict_router import router as predict_router
from app.api.v1.routes.root_router import router as root_router
from app.config.settings import settings


def create_app() -> FastAPI:
    """Create and configure the StoneSense AI FastAPI application.

    The application is intentionally built around a clean architecture
    boundary so that future ML and DL integrations can be introduced
    without changing the public API contract.
    """
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=settings.app_description,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    application.include_router(root_router)
    application.include_router(health_router)
    application.include_router(predict_router)
    application.include_router(assessment_router)

    return application


app = create_app()
