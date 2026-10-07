"""Testing configuration and fixtures setup module for pytest."""

import os
import sys
from pathlib import Path

# Ensure testing environment & JWT secret are set before application configuration is evaluated
os.environ["STONESENSE_ENV"] = "testing"
os.environ["ENVIRONMENT"] = "testing"
os.environ["STONESENSE_JWT_SECRET"] = "test-secret-key-stonesense-ai-phase4-super-secure"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "app"))

import pytest
from fastapi.testclient import TestClient

from app.config.settings import settings
settings.environment = "testing"
settings.jwt_secret_key = "test-secret-key-stonesense-ai-phase4-super-secure"

from app.main import app
from app.db.database import init_db, SessionLocal
from app.db.seed import run as seed_db
from app.core.security import create_access_token
from app.services.model_loader import model_loader


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Session fixture ensuring singleton model states and DB are initialized."""
    model_loader.load_all_models()
    init_db()
    seed_db()
    yield


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers_admin():
    token = create_access_token({
        "sub": "admin@stonesense.ai",
        "user_id": 1,
        "role": "admin",
        "hospital_id": None,
    })
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers_developer():
    token = create_access_token({
        "sub": "dev@stonesense.ai",
        "user_id": 2,
        "role": "developer",
        "hospital_id": None,
    })
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers_hospital_1():
    token = create_access_token({
        "sub": "hospital1@stonesense.ai",
        "user_id": 3,
        "role": "hospital_user",
        "hospital_id": 1,
    })
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers_hospital_2():
    token = create_access_token({
        "sub": "hospital2@stonesense.ai",
        "user_id": 4,
        "role": "hospital_user",
        "hospital_id": 2,
    })
    return {"Authorization": f"Bearer {token}"}
