from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.api.v1.routes.health_router import router as health_router
from app.api.v1.routes.predict_router import router as predict_router
from app.api.v1.routes.root_router import router as root_router
from app.api.v1.routes.model_router import router as model_router
from app.api.v1.routes.hospital import router as hospital_router
from app.api.v1.routes.developer import router as developer_router
from app.db.database import init_db
from app.config.settings import settings
from app.core.startup import load_models_on_startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model singletons once
    load_models_on_startup()
    init_db()
    yield


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
        lifespan=lifespan,
    )

    # Register CORS middleware BEFORE routers to support preflight OPTIONS requests
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000"
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(root_router)
    application.include_router(health_router, prefix="/api/v1")
    application.include_router(predict_router, prefix="/api/v1")
    application.include_router(model_router, prefix="/api/v1")
    application.include_router(hospital_router, prefix="/api/v1/hospital", tags=["hospital"])
    application.include_router(developer_router, prefix="/api/v1/developer", tags=["developer"])
    static_dir = Path(__file__).resolve().parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    application.mount("/static", StaticFiles(directory=static_dir), name="static")

    return application


app = create_app()

