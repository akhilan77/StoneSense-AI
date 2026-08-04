"""Unit tests for PredictionService and ModelLoader singleton classes."""

import pytest
from app.services.prediction_service import PredictionService
from app.services.model_loader import model_loader


def test_model_loader_singleton():
    """Verifies that ModelLoader maintains singleton state constraint."""
    from app.services.model_loader import ModelLoader
    loader1 = ModelLoader()
    loader2 = ModelLoader()
    assert loader1 is loader2
    assert loader1.dl_model is not None
    assert loader1.ml_model is not None


def test_prediction_service_risk():
    """Verifies prediction service outputs expected schemas on clinical dicts."""
    dummy_data = {
        "gravity": 1.015,
        "ph": 6.2,
        "osmo": 550,
        "cond": 22.0,
        "urea": 250,
        "calc": 4.5
    }
    res = PredictionService.predict_risk(dummy_data)
    assert "probability" in res
    assert "risk" in res
    assert res["risk"] in ["Low", "High"]
    assert 0.0 <= res["probability"] <= 1.0


def test_prediction_service_ct():
    """Verifies prediction service outputs on mock binary CT inputs."""
    # Create dummy black grayscale image bytes
    from PIL import Image
    import io
    img = Image.new("L", (224, 224), color=0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    res = PredictionService.predict_ct_image(img_bytes)
    assert "class" in res
    assert "confidence" in res
    assert res["class"] in ["Cyst", "Normal", "Stone", "Tumor"]
    assert 0.0 <= res["confidence"] <= 1.0
