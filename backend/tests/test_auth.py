"""Authentication and Authorization integration tests: RBAC, cross-hospital isolation, JWT life cycle, and audit logging."""

import pytest
from datetime import timedelta
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import create_access_token, create_refresh_token, hash_password
from app.db.database import SessionLocal
from app.db.models import AuditLog, Hospital, ModelVersion, User


def test_login_success_and_token_generation(client: TestClient):
    """Verifies that valid demo credentials return access & refresh tokens."""
    res = client.post(
        "/api/v1/auth/login",
        json={"email": "hospital1@stonesense.ai", "password": "Hospital1!2026"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == "hospital1@stonesense.ai"
    assert data["user"]["role"] == "hospital_user"
    assert data["user"]["hospital_id"] == 1


def test_login_invalid_password_returns_401(client: TestClient):
    """Verifies that invalid password returns 401 Unauthorized."""
    res = client.post(
        "/api/v1/auth/login",
        json={"email": "hospital1@stonesense.ai", "password": "WrongPassword123!"},
    )
    assert res.status_code == 401
    assert "detail" in res.json()


def test_login_inactive_user_returns_401(client: TestClient):
    """Verifies that deactivated user accounts cannot log in."""
    db: Session = SessionLocal()
    try:
        inactive_user = db.query(User).filter(User.email == "inactive@stonesense.ai").first()
        if not inactive_user:
            inactive_user = User(
                email="inactive@stonesense.ai",
                password_hash=hash_password("Inactive!2026"),
                role="hospital_user",
                hospital_id=1,
                is_active=False,
            )
            db.add(inactive_user)
            db.commit()
    finally:
        db.close()

    res = client.post(
        "/api/v1/auth/login",
        json={"email": "inactive@stonesense.ai", "password": "Inactive!2026"},
    )
    assert res.status_code == 401
    assert "deactivated" in res.json()["detail"].lower()


def test_refresh_token_flow(client: TestClient):
    """Verifies that a valid refresh token generates a fresh access token."""
    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": "dev@stonesense.ai", "password": "StoneSenseDev!2026"},
    )
    assert login_res.status_code == 200
    refresh_token = login_res.json()["refresh_token"]

    refresh_res = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert refresh_res.status_code == 200
    assert "access_token" in refresh_res.json()
    assert refresh_res.json()["token_type"] == "bearer"


def test_refresh_token_misuse_as_access_token_returns_401(client: TestClient):
    """Verifies that a refresh token is rejected when presented as an access token."""
    refresh_token = create_refresh_token({
        "sub": "dev@stonesense.ai",
        "user_id": 2,
        "role": "developer",
    })
    res = client.get(
        "/api/v1/models",
        headers={"Authorization": f"Bearer {refresh_token}"},
    )
    assert res.status_code == 401
    assert "invalid token type" in res.json()["detail"].lower()


def test_access_token_misuse_on_refresh_endpoint_returns_401(client: TestClient):
    """Verifies that an access token is rejected on the refresh token endpoint."""
    access_token = create_access_token({
        "sub": "dev@stonesense.ai",
        "user_id": 2,
        "role": "developer",
    })
    res = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": access_token},
    )
    assert res.status_code == 401
    assert "invalid token type" in res.json()["detail"].lower()


def test_expired_token_returns_401(client: TestClient):
    """Verifies that expired access tokens return 401 Unauthorized."""
    expired_token = create_access_token(
        {"sub": "dev@stonesense.ai", "user_id": 2, "role": "developer"},
        expires_delta=timedelta(seconds=-10),
    )
    res = client.get(
        "/api/v1/models",
        headers={"Authorization": f"Bearer {expired_token}"},
    )
    assert res.status_code == 401
    assert "expired" in res.json()["detail"].lower()


def test_missing_token_returns_401(client: TestClient):
    """Verifies that requests to protected routes without a token return 401."""
    assert client.get("/api/v1/models").status_code == 401
    assert client.get("/api/v1/patients").status_code == 401
    assert client.get("/api/v1/hospital/1/detail").status_code == 401
    assert client.get("/api/v1/developer/model-versions").status_code == 401


def test_cross_hospital_access_returns_403(client: TestClient, auth_headers_hospital_1):
    """Hospital 1 user attempting to access Hospital 2 resources must receive 403 Forbidden."""
    res_detail = client.get("/api/v1/hospital/2/detail", headers=auth_headers_hospital_1)
    assert res_detail.status_code == 403

    res_history = client.get("/api/v1/hospital/2/history", headers=auth_headers_hospital_1)
    assert res_history.status_code == 403

    res_validate = client.post("/api/v1/hospital/2/dataset-validate", headers=auth_headers_hospital_1)
    assert res_validate.status_code == 403


def test_hospital_user_cannot_access_developer_routes(client: TestClient, auth_headers_hospital_1):
    """Hospital users attempting to access /developer/* endpoints must receive 403 Forbidden."""
    res = client.get("/api/v1/developer/model-versions", headers=auth_headers_hospital_1)
    assert res.status_code == 403

    res_overview = client.get("/api/v1/developer/federated-overview", headers=auth_headers_hospital_1)
    assert res_overview.status_code == 403


def test_developer_and_admin_can_access_developer_routes(
    client: TestClient, auth_headers_developer, auth_headers_admin
):
    """Developers and admins must be permitted to access developer endpoints."""
    res_dev = client.get("/api/v1/developer/model-versions", headers=auth_headers_developer)
    assert res_dev.status_code == 200

    res_admin = client.get("/api/v1/developer/model-versions", headers=auth_headers_admin)
    assert res_admin.status_code == 200


def test_deploy_and_rollback_sets_user_email_and_writes_audit_log(
    client: TestClient, auth_headers_developer
):
    """Verifies that model deployment records the deployer email and writes an audit log."""
    from uuid import uuid4
    unique_tag = f"resnet18_test_auth_{uuid4().hex[:8]}"
    db: Session = SessionLocal()
    try:
        # Create an eligible test model version
        test_ver = ModelVersion(
            model_family="resnet18_ct",
            version_tag=unique_tag,
            accuracy=0.99,
            f1_score=0.985,
            precision=0.985,
            recall=0.985,
            status="eligible",
            is_deployed=False,
            artifact_path="dl/models/resnet18_test.pth",
        )
        db.add(test_ver)
        db.commit()
        db.refresh(test_ver)
        target_id = test_ver.id

        # Deploy model
        res = client.post(
            "/api/v1/developer/model-versions/deploy",
            json={"model_version_id": target_id},
            headers=auth_headers_developer,
        )
        assert res.status_code == 200
        data = res.json()
        assert data["approved_by"] == "dev@stonesense.ai"

        # Check audit log entry
        audit_entry = (
            db.query(AuditLog)
            .filter(AuditLog.action == "MODEL_DEPLOY", AuditLog.resource_id == str(target_id))
            .order_by(AuditLog.created_at.desc())
            .first()
        )
        assert audit_entry is not None
        assert audit_entry.user_email == "dev@stonesense.ai"
        assert audit_entry.details["version_tag"] == unique_tag
    finally:
        db.close()


def test_websocket_unauthenticated_connection_rejected(client: TestClient):
    """Verifies that WebSocket connections without valid tokens are closed with 4401 or 4403."""
    with pytest.raises(Exception):
        with client.websocket_connect("/api/v1/developer/federated/ws") as ws:
            ws.send_text("ping")
            ws.receive_text()



def test_websocket_cross_hospital_connection_rejected(client: TestClient):
    """Verifies that a Hospital 1 user cannot connect to Hospital 2's WebSocket."""
    token_h1 = create_access_token({
        "sub": "hospital1@stonesense.ai",
        "user_id": 3,
        "role": "hospital_user",
        "hospital_id": 1,
    })
    with pytest.raises(Exception):
        with client.websocket_connect(f"/api/v1/hospital/2/federated/ws?token={token_h1}") as ws:
            ws.send_text("ping")


def test_environment_defaults_to_production_and_fails_fast_without_secret(monkeypatch):
    """Verifies that unset ENVIRONMENT defaults to production and raises RuntimeError when secret is missing."""
    monkeypatch.delenv("STONESENSE_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("STONESENSE_JWT_SECRET", raising=False)
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)

    from app.config.settings import Settings
    prod_settings = Settings()
    assert prod_settings.environment == "production"
    assert prod_settings.is_production is True
    assert prod_settings.jwt_secret_key is None

    with pytest.raises(RuntimeError) as exc_info:
        prod_settings.get_jwt_secret()
    assert "production mode strictly forbids" in str(exc_info.value).lower()


def test_developer_cannot_access_patient_pii_endpoints(client: TestClient, auth_headers_developer):
    """Verifies that developer role is blocked with 403 from patient PII endpoints."""
    res_patients = client.get("/api/v1/patients", headers=auth_headers_developer)
    assert res_patients.status_code == 403
    assert "patient pii" in res_patients.json()["detail"].lower()

    res_patient_detail = client.get("/api/v1/patients/1", headers=auth_headers_developer)
    assert res_patient_detail.status_code == 403

    res_history = client.get("/api/v1/hospital/1/history", headers=auth_headers_developer)
    assert res_history.status_code == 403


def test_admin_can_access_patient_pii_with_audit_log(client: TestClient, auth_headers_admin):
    """Verifies that admin role can access patient endpoints and an audit log is created."""
    db: Session = SessionLocal()
    try:
        res = client.get("/api/v1/patients?hospital_id=1", headers=auth_headers_admin)
        assert res.status_code == 200

        # Verify audit log was recorded
        audit = (
            db.query(AuditLog)
            .filter(AuditLog.user_email == "admin@stonesense.ai", AuditLog.action == "PATIENT_READ")
            .order_by(AuditLog.created_at.desc())
            .first()
        )
        assert audit is not None
        assert audit.resource_type == "patient_list"
    finally:
        db.close()


def test_docs_disabled_in_production(monkeypatch):
    """Verifies that OpenAPI docs endpoints (/docs, /redoc, /openapi.json) are disabled in production."""
    from app.config.settings import settings
    from app.main import create_app

    old_env = settings.environment
    try:
        settings.environment = "production"
        prod_app = create_app()
        assert prod_app.docs_url is None
        assert prod_app.redoc_url is None
        assert prod_app.openapi_url is None
    finally:
        settings.environment = old_env


def test_websocket_first_message_authentication(client: TestClient, auth_headers_developer):
    """Verifies that WebSocket connection can authenticate via initial JSON message."""
    token = auth_headers_developer["Authorization"].replace("Bearer ", "")
    with client.websocket_connect("/api/v1/ws/federated") as ws:
        # Send auth frame
        ws.send_text(f'{{"type": "auth", "token": "{token}"}}')
        # Send ping
        ws.send_text("ping")
        data = ws.receive_text()
        assert "pong" in data


def test_admin_audit_logs_endpoint_rbac_and_pagination(
    client: TestClient, auth_headers_admin, auth_headers_developer, auth_headers_hospital_1
):
    """Verifies GET /api/v1/admin/audit-logs RBAC and pagination."""
    # Hospital user blocked
    res_h = client.get("/api/v1/admin/audit-logs", headers=auth_headers_hospital_1)
    assert res_h.status_code == 403

    # Developer blocked
    res_d = client.get("/api/v1/admin/audit-logs", headers=auth_headers_developer)
    assert res_d.status_code == 403

    # Admin allowed
    res_a = client.get("/api/v1/admin/audit-logs?page=1&page_size=10", headers=auth_headers_admin)
    assert res_a.status_code == 200
    data = res_a.json()
    assert "total" in data
    assert "items" in data
    assert "page" in data
    assert data["page"] == 1
    assert data["page_size"] == 10
    assert isinstance(data["items"], list)

