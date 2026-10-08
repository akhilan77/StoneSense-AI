"""Integration tests for multi-model DL inference and API endpoints."""

import io
import pytest
from PIL import Image
from backend.app.services.prediction_service import PredictionService
from backend.app.services.model_loader import model_loader
from dl.models.registry import dl_registry


@pytest.fixture(scope="module", autouse=True)
def setup_models():
    """Ensure all available DL and ML models are loaded in registry and loader."""
    dl_registry.load_all_models()
    model_loader.load_all_models()
    yield


def create_dummy_image_bytes() -> bytes:
    """Creates a sample 224x224 RGB image in memory for testing."""
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_resnet18_inference_parity():
    """Verify that ResNet18 inference succeeds and returns standard contract keys."""
    img_bytes = create_dummy_image_bytes()
    res = PredictionService.predict_ct_image(img_bytes, model_id="resnet18")
    assert res["class"] in ["Cyst", "Normal", "Stone", "Tumor"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert res["model_id"] == "resnet18"
    assert res["model_name"] == "ResNet18"
    assert "probabilities" in res


def test_default_inference_uses_resnet18():
    """Verify that omitting model_id defaults strictly to ResNet18."""
    img_bytes = create_dummy_image_bytes()
    res = PredictionService.predict_ct_image(img_bytes)
    assert res["model_id"] == "resnet18"
    assert res["model_name"] == "ResNet18"


def test_pending_dinov3_raises_runtime_error():
    """Verify that calling predict on pending DINOv3 raises RuntimeError."""
    img_bytes = create_dummy_image_bytes()
    with pytest.raises(RuntimeError) as exc_info:
        PredictionService.predict_ct_image(img_bytes, model_id="dinov3")
    assert "not ready" in str(exc_info.value).lower() or "pending" in str(exc_info.value).lower()


def test_pending_qknn_raises_runtime_error():
    """Verify that calling predict on pending QKNN raises RuntimeError."""
    img_bytes = create_dummy_image_bytes()
    with pytest.raises(RuntimeError) as exc_info:
        PredictionService.predict_ct_image(img_bytes, model_id="qknn")
    assert "not ready" in str(exc_info.value).lower() or "pending" in str(exc_info.value).lower()


def test_api_get_dl_models(auth_client):
    """Verify that GET /api/v1/models/dl returns all 4 DL models with correct metadata."""
    response = auth_client.get("/api/v1/models/dl")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    models = data["models"]
    assert len(models) == 4

    ids = [m["id"] for m in models]
    assert ids == ["resnet18", "yolo26", "dinov3", "qknn"]

    # Verify YOLO26 benchmark metrics and status in API response
    yolo_meta = next(m for m in models if m["id"] == "yolo26")
    assert yolo_meta["name"] == "YOLO26"
    assert yolo_meta["accuracy"] == pytest.approx(0.921883, abs=1e-3)
    assert yolo_meta["macro_f1"] == pytest.approx(0.904705, abs=1e-3)
    assert yolo_meta["status"] == "PENDING_WEIGHTS"
    assert yolo_meta["is_ready"] is False

    # Verify DINOv3 and QKNN status
    dino_meta = next(m for m in models if m["id"] == "dinov3")
    assert dino_meta["status"] == "PENDING_WEIGHTS"

    qknn_meta = next(m for m in models if m["id"] == "qknn")
    assert qknn_meta["status"] == "PENDING_WEIGHTS"


def test_api_predict_image_backward_compatibility(auth_client):
    """Verify that POST /api/v1/predict/image with only an image file defaults to ResNet18."""
    img_bytes = create_dummy_image_bytes()
    files = {"image": ("test_scan.png", img_bytes, "image/png")}
    response = auth_client.post("/api/v1/predict/image", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["class_name"] in ["Cyst", "Normal", "Stone", "Tumor"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["model_id"] == "resnet18"
    assert data["model_name"] == "ResNet18"
    assert "gradcam" in data


def test_api_predict_image_pending_yolo26_returns_503(auth_client):
    """Verify that selecting YOLO26 when weights are pending returns a clean 503 response."""
    img_bytes = create_dummy_image_bytes()
    files = {"image": ("test_scan.png", img_bytes, "image/png")}
    data = {"model_family": "yolo26"}
    response = auth_client.post("/api/v1/predict/image", files=files, data=data)
    assert response.status_code == 503
    err_detail = response.json().get("detail", "")
    assert "YOLO26" in err_detail
    assert "PENDING_WEIGHTS" in err_detail or "unavailable" in err_detail.lower()


def test_api_predict_image_pending_dinov3_returns_503(auth_client):
    """Verify that selecting DINOv3 when weights are pending returns a clean 503 response."""
    img_bytes = create_dummy_image_bytes()
    files = {"image": ("test_scan.png", img_bytes, "image/png")}
    data = {"model_family": "dinov3"}
    response = auth_client.post("/api/v1/predict/image", files=files, data=data)
    assert response.status_code == 503
    err_detail = response.json().get("detail", "")
    assert "DINOv3" in err_detail
    assert "PENDING_WEIGHTS" in err_detail or "unavailable" in err_detail.lower()


def test_api_predict_image_pending_qknn_returns_503(auth_client):
    """Verify that selecting QKNN when weights are pending returns a clean 503 response."""
    img_bytes = create_dummy_image_bytes()
    files = {"image": ("test_scan.png", img_bytes, "image/png")}
    data = {"model_family": "qknn"}
    response = auth_client.post("/api/v1/predict/image", files=files, data=data)
    assert response.status_code == 503
    err_detail = response.json().get("detail", "")
    assert "QKNN" in err_detail
    assert "PENDING_WEIGHTS" in err_detail or "unavailable" in err_detail.lower()
