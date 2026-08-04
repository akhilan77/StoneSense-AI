"""Testing configuration and fixtures setup module for pytest."""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "app"))

from app.main import app
from app.services.model_loader import model_loader


@pytest.fixture(scope="session", autouse=True)
def init_models():
    """Session fixture ensuring singleton model states are initialized."""
    model_loader.load_all_models()
    yield


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as c:
        yield c
