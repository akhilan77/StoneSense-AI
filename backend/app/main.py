from contextlib import asynccontextmanager
import json
import asyncio
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.routes.auth import router as auth_router
from app.api.v1.routes.admin import router as admin_router
from app.api.v1.routes.health_router import router as health_router
from app.api.v1.routes.predict_router import router as predict_router
from app.api.v1.routes.root_router import router as root_router
from app.api.v1.routes.model_router import router as model_router
from app.api.v1.routes.hospital import router as hospital_router
from app.api.v1.routes.developer import router as developer_router
from app.api.v1.routes.patients import router as patients_router
from app.db.database import init_db
from app.config.settings import settings
from app.core.security import decode_token
from app.core.startup import load_models_on_startup
from app.services.ws_manager import ws_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate configuration on startup
    _ = settings.get_jwt_secret()
    load_models_on_startup()
    init_db()
    yield


async def _authenticate_websocket(websocket: WebSocket) -> dict:
    """Extracts and validates JWT access token from Authorization header or query param."""
    auth_header = websocket.headers.get("authorization") or websocket.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]
        return decode_token(token, expected_type="access")

    token = websocket.query_params.get("token")
    if token:
        return decode_token(token, expected_type="access")

    raise ValueError("Missing authentication token")


def create_app() -> FastAPI:
    """Create and configure the StoneSense AI FastAPI application."""
    docs_url = None if settings.is_production else "/docs"
    redoc_url = None if settings.is_production else "/redoc"
    openapi_url = None if settings.is_production else "/openapi.json"

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=settings.app_description,
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
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
    application.include_router(auth_router, prefix="/api/v1")
    application.include_router(admin_router, prefix="/api/v1")
    application.include_router(health_router, prefix="/api/v1")
    application.include_router(predict_router, prefix="/api/v1")
    application.include_router(model_router, prefix="/api/v1")
    application.include_router(hospital_router, prefix="/api/v1/hospital", tags=["hospital"])
    application.include_router(hospital_router, prefix="/api/v1/hospitals", tags=["hospitals"])
    application.include_router(developer_router, prefix="/api/v1/developer", tags=["developer"])
    application.include_router(patients_router, prefix="/api/v1")

    @application.websocket("/api/v1/ws/federated")
    @application.websocket("/api/v1/developer/federated/ws")
    @application.websocket("/api/v1/federated/ws")
    async def federated_websocket(websocket: WebSocket):
        claims = None
        already_accepted = False
        try:
            claims = await _authenticate_websocket(websocket)
        except ValueError:
            # Fall back to first-message authentication
            await websocket.accept()
            already_accepted = True
            try:
                raw_msg = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                parsed = json.loads(raw_msg)
                token = parsed.get("token") or (parsed.get("data", {}).get("token") if isinstance(parsed.get("data"), dict) else None)
                if not token:
                    await websocket.close(code=4401, reason="Unauthorized: Missing token in first message")
                    return
                claims = decode_token(token, expected_type="access")
            except Exception:
                await websocket.close(code=4401, reason="Unauthorized: Missing or invalid token")
                return

        role = claims.get("role")
        if role not in ("developer", "admin"):
            if not already_accepted:
                await websocket.close(code=4403, reason="Forbidden: Insufficient role permissions")
            else:
                await websocket.close(code=4403, reason="Forbidden: Insufficient role permissions")
            return

        if not already_accepted:
            await ws_manager.connect(websocket)
        else:
            await ws_manager.register(websocket)

        try:
            while True:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text('{"event": "pong"}')
        except (WebSocketDisconnect, Exception):
            await ws_manager.disconnect(websocket)

    @application.websocket("/api/v1/hospital/{hospital_id}/federated/ws")
    async def hospital_federated_websocket(websocket: WebSocket, hospital_id: str):
        claims = None
        already_accepted = False
        try:
            claims = await _authenticate_websocket(websocket)
        except ValueError:
            # Fall back to first-message authentication
            await websocket.accept()
            already_accepted = True
            try:
                raw_msg = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                parsed = json.loads(raw_msg)
                token = parsed.get("token") or (parsed.get("data", {}).get("token") if isinstance(parsed.get("data"), dict) else None)
                if not token:
                    await websocket.close(code=4401, reason="Unauthorized: Missing token in first message")
                    return
                claims = decode_token(token, expected_type="access")
            except Exception:
                await websocket.close(code=4401, reason="Unauthorized: Missing or invalid token")
                return

        role = claims.get("role")
        user_hospital_id = claims.get("hospital_id")
        hospital_codes = {"1": "HOSP-001", "2": "HOSP-002", "3": "HOSP-003"}
        target_code = hospital_codes.get(hospital_id, hospital_id)

        # Scoping validation for hospital users
        if role == "hospital_user":
            user_hcode = hospital_codes.get(str(user_hospital_id), str(user_hospital_id))
            if str(user_hospital_id) != str(hospital_id) and user_hcode != target_code:
                await websocket.close(code=4403, reason="Forbidden: Cross-hospital WebSocket access not permitted")
                return
        elif role not in ("developer", "admin"):
            await websocket.close(code=4403, reason="Forbidden: Insufficient role permissions")
            return

        if not already_accepted:
            await ws_manager.connect(websocket, hospital_id=target_code)
        else:
            await ws_manager.register(websocket, hospital_id=target_code)

        try:
            while True:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text('{"event": "pong"}')
        except (WebSocketDisconnect, Exception):
            await ws_manager.disconnect(websocket)


    static_dir = Path(__file__).resolve().parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    application.mount("/static", StaticFiles(directory=static_dir), name="static")

    return application


app = create_app()

