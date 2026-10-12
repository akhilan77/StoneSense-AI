"""Root pytest configuration to guarantee test isolation from production artifacts and database."""

import os
import sys
import tempfile
from pathlib import Path

# Protect repository stonesense.db and tracked checkpoints
_TEST_TEMP_DIR = Path(tempfile.gettempdir()) / f"stonesense_test_run_{os.getpid()}"
_TEST_TEMP_DIR.mkdir(parents=True, exist_ok=True)
_TEST_DB_PATH = _TEST_TEMP_DIR / "stonesense_test.db"
_TEST_MODELS_DIR = _TEST_TEMP_DIR / "models"

os.environ["STONESENSE_DATABASE_URL"] = f"sqlite:///{_TEST_DB_PATH.resolve()}"
os.environ["STONESENSE_MODELS_DIR"] = str(_TEST_MODELS_DIR.resolve())
os.environ["STONESENSE_ENV"] = "testing"
os.environ["ENVIRONMENT"] = "testing"
os.environ["STONESENSE_JWT_SECRET"] = "test-secret-key-stonesense-ai-phase4-super-secure"

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "dl" / "federated"))

import pytest


@pytest.fixture(scope="session", autouse=True)
def global_test_isolation_fixture():
    try:
        from app.db.database import set_database_url
        set_database_url(f"sqlite:///{_TEST_DB_PATH.resolve()}")
    except Exception:
        pass
    try:
        from dl.federated.model_manager import model_manager
        model_manager.set_models_dir(_TEST_MODELS_DIR)
    except Exception:
        pass
    yield
