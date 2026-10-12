import os
import sys
import tempfile
from pathlib import Path

# Redirect DB and models to temporary directory to protect stonesense.db and tracked checkpoints
_TEST_TEMP_DIR = Path(tempfile.gettempdir()) / f"stonesense_test_run_{os.getpid()}"
_TEST_TEMP_DIR.mkdir(parents=True, exist_ok=True)
_TEST_DB_PATH = _TEST_TEMP_DIR / "stonesense_test.db"
_TEST_MODELS_DIR = _TEST_TEMP_DIR / "models"

os.environ["STONESENSE_DATABASE_URL"] = f"sqlite:///{_TEST_DB_PATH.resolve()}"
os.environ["STONESENSE_MODELS_DIR"] = str(_TEST_MODELS_DIR.resolve())
os.environ["STONESENSE_ENV"] = "testing"
os.environ["ENVIRONMENT"] = "testing"
os.environ["STONESENSE_JWT_SECRET"] = "test-secret-key-stonesense-ai-phase4-super-secure"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "dl" / "federated"))

import pytest
from fastapi.testclient import TestClient

from app.config.settings import settings
settings.environment = "testing"
settings.jwt_secret_key = "test-secret-key-stonesense-ai-phase4-super-secure"

from app.main import app
from app.db.database import init_db, set_database_url
from app.db.seed import run as seed_db
from app.core.security import create_access_token


@pytest.fixture(scope="session", autouse=True)
def setup_test_suite_db():
    set_database_url(f"sqlite:///{_TEST_DB_PATH.resolve()}")
    try:
        from dl.federated.model_manager import model_manager
        model_manager.set_models_dir(_TEST_MODELS_DIR)
    except Exception:
        pass
    init_db()
    seed_db()
    yield


@pytest.fixture(scope="session")
def admin_token():
    return create_access_token(data={"sub": "admin@stonesense.ai", "role": "admin", "hospital_id": None})


@pytest.fixture(scope="session")
def dev_token():
    return create_access_token(data={"sub": "dev@stonesense.ai", "role": "developer", "hospital_id": None})


@pytest.fixture(scope="session")
def hospital1_token():
    return create_access_token(data={"sub": "hospital1@stonesense.ai", "role": "hospital_user", "hospital_id": 1})


@pytest.fixture(scope="session")
def admin_headers(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture(scope="session")
def dev_headers(dev_token):
    return {"Authorization": f"Bearer {dev_token}"}


@pytest.fixture(scope="session")
def hospital1_headers(hospital1_token):
    return {"Authorization": f"Bearer {hospital1_token}"}


@pytest.fixture(scope="session")
def auth_client(admin_headers):
    with TestClient(app, headers=admin_headers) as c:
        yield c

