"""Tests for DL Model Registry and multi-model abstractions."""

import pytest
from pathlib import Path
from dl.models.base import BaseCTModel, DLModelStatus
from dl.models.registry import DLModelRegistry, dl_registry
from dl.models.wrappers.resnet18_wrapper import ResNet18Wrapper
from dl.models.wrappers.yolo26_wrapper import YOLO26Wrapper
from dl.models.wrappers.dinov3_wrapper import DINOv3Wrapper
from dl.models.wrappers.qknn_wrapper import QKNNWrapper


def test_dl_registry_initialization():
    """Verify that all 4 models are registered with their expected IDs and families."""
    reg = DLModelRegistry()
    models = reg.list_models()
    assert len(models) == 4

    ids = [m["id"] for m in models]
    assert "resnet18" in ids
    assert "yolo26" in ids
    assert "dinov3" in ids
    assert "qknn" in ids


def test_resnet18_wrapper_loads_or_pending():
    """Verify that ResNet18 wrapper resolves weights and loads if available."""
    reg = DLModelRegistry()
    reg.load_all_models()
    r18 = reg.get_model("resnet18")
    assert isinstance(r18, ResNet18Wrapper)
    assert r18.model_id == "resnet18"
    assert r18.model_family == "resnet18_ct"
    # If weights exist in workspace, it should be READY
    pth_path = Path("dl/models/kidney_resnet18.pth")
    if pth_path.exists():
        assert r18.status == DLModelStatus.READY
        assert r18.is_ready is True
        assert r18.model is not None


def test_yolo26_wrapper_discovery_and_metadata():
    """Verify that YOLO26 wrapper resolves metadata and handles status correctly."""
    reg = DLModelRegistry()
    reg.load_all_models()
    yolo = reg.get_model("yolo26")
    assert isinstance(yolo, YOLO26Wrapper)
    assert yolo.model_id == "yolo26"
    assert yolo.model_family == "yolo26_ct"

    meta = yolo.get_metadata()
    assert meta["id"] == "yolo26"
    assert meta["name"] == "YOLO26"
    # When best.pt is not yet present on disk, status must be PENDING_WEIGHTS
    best_pt = Path("dl/models/ct/yolo26/best.pt")
    if not best_pt.exists():
        assert yolo.status == DLModelStatus.PENDING_WEIGHTS
        assert yolo.is_ready is False
    # Verified benchmark metrics should be present from metrics.json
    assert meta["accuracy"] == pytest.approx(0.921883, abs=1e-3)
    assert meta["macro_f1"] == pytest.approx(0.904705, abs=1e-3)
    assert meta["stone_recall"] == pytest.approx(0.942028, abs=1e-3)
    assert meta["tumor_recall"] == pytest.approx(0.991253, abs=1e-3)


def test_dinov3_reports_pending_weights():
    """Verify that DINOv3 reports PENDING_WEIGHTS when artifact weights are absent."""
    reg = DLModelRegistry()
    reg.load_all_models()
    dino = reg.get_model("dinov3")
    assert isinstance(dino, DINOv3Wrapper)
    assert dino.model_id == "dinov3"
    assert dino.status == DLModelStatus.PENDING_WEIGHTS
    assert dino.is_ready is False


def test_qknn_reports_pending_weights():
    """Verify that QKNN reports PENDING_WEIGHTS when artifact weights are absent."""
    reg = DLModelRegistry()
    reg.load_all_models()
    qknn = reg.get_model("qknn")
    assert isinstance(qknn, QKNNWrapper)
    assert qknn.model_id == "qknn"
    assert qknn.status == DLModelStatus.PENDING_WEIGHTS
    assert qknn.is_ready is False


def test_alias_resolution():
    """Verify that aliases like 'resnet18_ct', 'yolo', etc. resolve correctly."""
    assert dl_registry.resolve_model_id("resnet18_ct") == "resnet18"
    assert dl_registry.resolve_model_id("resnet") == "resnet18"
    assert dl_registry.resolve_model_id("yolo26_ct") == "yolo26"
    assert dl_registry.resolve_model_id("yolo") == "yolo26"
    assert dl_registry.resolve_model_id("dinov3_ct") == "dinov3"
    assert dl_registry.resolve_model_id("qknn_ct") == "qknn"
    assert dl_registry.resolve_model_id(None) == "resnet18"


def test_unknown_model_raises_keyerror():
    """Verify that unknown model IDs raise KeyError."""
    with pytest.raises(KeyError):
        dl_registry.get_model("unknown_future_model")
