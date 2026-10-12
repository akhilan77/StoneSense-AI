"""Tests for CT image upload validation, security hardening, and OOD heuristic gate."""

import io
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config.settings import settings


def _make_image_bytes(format="PNG", size=(224, 224), color=128, mode="L"):
    """Generates synthetic image bytes for testing."""
    img = Image.new(mode, size, color=color)
    # Add non-uniform gradient so image has non-zero standard deviation
    if size[0] >= 32 and size[1] >= 32 and mode == "L":
        for x in range(min(50, size[0])):
            for y in range(min(50, size[1])):
                img.putpixel((x, y), (x * y) % 255)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def test_valid_png_upload(client: TestClient, auth_headers_hospital_1):
    """Valid PNG image upload is accepted (HTTP 200) with status='accepted'."""
    img_bytes = _make_image_bytes(format="PNG", size=(224, 224))
    files = {"image": ("axial_ct.png", img_bytes, "image/png")}
    data = {"hospital_id": 1}

    res = client.post("/api/v1/predict/image", files=files, data=data, headers=auth_headers_hospital_1)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "accepted"
    assert body["class_name"] in ["Cyst", "Normal", "Stone", "Tumor"]
    assert "probabilities" in body
    assert "ood_metrics" in body


def test_valid_jpeg_upload(client: TestClient, auth_headers_hospital_1):
    """Valid JPEG image upload is accepted (HTTP 200) with status='accepted'."""
    img_bytes = _make_image_bytes(format="JPEG", size=(224, 224), mode="RGB")
    files = {"image": ("axial_ct.jpg", img_bytes, "image/jpeg")}
    data = {"hospital_id": 1}

    res = client.post("/api/v1/predict/image", files=files, data=data, headers=auth_headers_hospital_1)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "accepted"


def test_oversized_file_rejected_413(client: TestClient, auth_headers_hospital_1):
    """Upload exceeding max byte limit is rejected with HTTP 413 Payload Too Large."""
    # Temporarily set max upload bytes to 100 KB for test
    original_max = settings.ct_upload_max_bytes
    settings.ct_upload_max_bytes = 100 * 1024
    try:
        oversized_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * (120 * 1024)
        files = {"image": ("huge_ct.png", oversized_bytes, "image/png")}
        data = {"hospital_id": 1}

        res = client.post("/api/v1/predict/image", files=files, data=data, headers=auth_headers_hospital_1)
        assert res.status_code == 413
        assert "exceeds maximum allowed upload size" in res.json()["detail"].lower()
    finally:
        settings.ct_upload_max_bytes = original_max


def test_wrong_magic_bytes_with_png_extension_415(client: TestClient, auth_headers_hospital_1):
    """Text / HTML file with a .png extension is rejected with HTTP 415 Unsupported Media Type."""
    fake_png_bytes = b"This is a plain text file pretending to be a medical CT scan."
    files = {"image": ("fake_ct.png", fake_png_bytes, "image/png")}
    data = {"hospital_id": 1}

    res = client.post("/api/v1/predict/image", files=files, data=data, headers=auth_headers_hospital_1)
    assert res.status_code == 415
    assert "unsupported media type" in res.json()["detail"].lower()


def test_wrong_magic_bytes_with_jpg_extension_415(client: TestClient, auth_headers_hospital_1):
    """PDF file with a .jpg extension is rejected with HTTP 415."""
    fake_jpg_bytes = b"%PDF-1.4 simulated pdf document"
    files = {"image": ("document.jpg", fake_jpg_bytes, "image/jpeg")}
    data = {"hospital_id": 1}

    res = client.post("/api/v1/predict/image", files=files, data=data, headers=auth_headers_hospital_1)
    assert res.status_code == 415


def test_zero_byte_file_rejected_422(client: TestClient, auth_headers_hospital_1):
    """Zero-byte file upload is rejected with HTTP 422 Unprocessable Entity."""
    empty_bytes = b""
    files = {"image": ("empty.png", empty_bytes, "image/png")}
    data = {"hospital_id": 1}

    res = client.post("/api/v1/predict/image", files=files, data=data, headers=auth_headers_hospital_1)
    assert res.status_code == 422
    assert "0 bytes" in res.json()["detail"]


def test_truncated_corrupt_file_rejected_422(client: TestClient, auth_headers_hospital_1):
    """Truncated file that starts with JPEG magic bytes but lacks image data is rejected with 422."""
    corrupt_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 20
    files = {"image": ("corrupted.jpg", corrupt_bytes, "image/jpeg")}
    data = {"hospital_id": 1}

    res = client.post("/api/v1/predict/image", files=files, data=data, headers=auth_headers_hospital_1)
    assert res.status_code == 422
    assert "corrupted" in res.json()["detail"].lower()


def test_decompression_bomb_rejected_413(client: TestClient, auth_headers_hospital_1):
    """Image exceeding MAX_IMAGE_PIXELS triggers DecompressionBombError and returns 413."""
    original_max_pixels = settings.ct_upload_max_pixels
    # Set limit below test image size (e.g. 1,000 pixels)
    settings.ct_upload_max_pixels = 1000
    try:
        # A 224x224 image has 50,176 pixels, which exceeds 1,000
        img_bytes = _make_image_bytes(format="PNG", size=(224, 224))
        files = {"image": ("bomb.png", img_bytes, "image/png")}
        data = {"hospital_id": 1}

        res = client.post("/api/v1/predict/image", files=files, data=data, headers=auth_headers_hospital_1)
        assert res.status_code == 413
        assert "decompression bomb" in res.json()["detail"].lower()
    finally:
        settings.ct_upload_max_pixels = original_max_pixels


def test_degenerate_dimensions_too_small_422(client: TestClient, auth_headers_hospital_1):
    """Image dimensions smaller than minimum threshold (32x32) are rejected with 422."""
    tiny_img_bytes = _make_image_bytes(format="PNG", size=(16, 16))
    files = {"image": ("tiny.png", tiny_img_bytes, "image/png")}
    data = {"hospital_id": 1}

    res = client.post("/api/v1/predict/image", files=files, data=data, headers=auth_headers_hospital_1)
    assert res.status_code == 422
    assert "too small" in res.json()["detail"].lower()


def test_extreme_aspect_ratio_rejected_422(client: TestClient, auth_headers_hospital_1):
    """Image with extreme non-axial aspect ratio (e.g. 500x50 = 10:1) is rejected with 422."""
    distorted_bytes = _make_image_bytes(format="PNG", size=(500, 50))
    files = {"image": ("strip.png", distorted_bytes, "image/png")}
    data = {"hospital_id": 1}

    res = client.post("/api/v1/predict/image", files=files, data=data, headers=auth_headers_hospital_1)
    assert res.status_code == 422
    assert "aspect ratio" in res.json()["detail"].lower()


def test_near_constant_image_warning(client: TestClient, auth_headers_hospital_1):
    """Solid uniform image has low pixel variance and produces a near-constant warning."""
    # Uniform solid gray (standard deviation = 0.0)
    img = Image.new("L", (224, 224), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    solid_bytes = buf.getvalue()

    files = {"image": ("solid.png", solid_bytes, "image/png")}
    data = {"hospital_id": 1}

    res = client.post("/api/v1/predict/image", files=files, data=data, headers=auth_headers_hospital_1)
    assert res.status_code == 200
    body = res.json()
    assert any("Near-constant image detected" in w for w in body.get("warnings", []))


def test_low_confidence_ood_gate_fires(client: TestClient, auth_headers_hospital_1):
    """When model outputs low confidence or high entropy, the OOD gate returns status='needs_review'."""
    img_bytes = _make_image_bytes(format="PNG", size=(224, 224))
    files = {"image": ("ambiguous.png", img_bytes, "image/png")}
    data = {"hospital_id": 1}

    mock_prediction = {
        "class": "Stone",
        "class_name": "Stone",
        "confidence": 0.35,  # below 0.50 gate threshold
        "model_id": "resnet18",
        "model_name": "ResNet18",
        "model_family": "resnet18_ct",
        "probabilities": {
            "Cyst": 0.25,
            "Normal": 0.20,
            "Stone": 0.35,
            "Tumor": 0.20,
        },
        "inference_time_sec": 0.12,
    }

    with patch("app.services.prediction_service.PredictionService.predict_ct_image", return_value=mock_prediction):
        res = client.post("/api/v1/predict/image", files=files, data=data, headers=auth_headers_hospital_1)
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "needs_review"
        assert body["class_name"] == "needs_review"
        assert body["review_reason"] is not None
        assert "below safety gate" in body["review_reason"]
        assert body["probabilities"]["Stone"] == 0.35
        assert body["ood_metrics"]["confidence"] == 0.35
