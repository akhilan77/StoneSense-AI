"""Focused tests for QKNN artifact loading, preprocessing, and predictions."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dl.models.wrappers.qknn_wrapper import QKNNWrapper


class _Scaler:
    def __init__(self, n_features_in, offset):
        self.n_features_in_ = n_features_in
        self.offset = offset

    def transform(self, values):
        return values + self.offset


class _PCA:
    def __init__(self, n_features_in, components):
        self.n_features_in_ = n_features_in
        self.components = components

    def transform(self, values):
        return values[:, :self.components] * 2


def _fake_pennylane(monkeypatch):
    def qnode(_device):
        def decorate(_circuit):
            def fidelity(x1, x2):
                similarity = np.prod(np.cos((x1 - x2) / 2) ** 2)
                return np.array([similarity, 1 - similarity])
            return fidelity
        return decorate

    fake_qml = SimpleNamespace(
        device=lambda *_args, **_kwargs: object(),
        qnode=qnode,
        AngleEmbedding=lambda *_args, **_kwargs: None,
        adjoint=lambda operation: operation,
        probs=lambda **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "pennylane", fake_qml)


def test_transform_embedding_supports_legacy_scaler_before_pca():
    wrapper = QKNNWrapper()
    wrapper.pca_pipeline = {
        "scaler": _Scaler(n_features_in=3, offset=1),
        "pca": _PCA(n_features_in=3, components=2),
    }

    result = wrapper._transform_embedding(np.array([1, 2, 3], dtype=np.float32))

    np.testing.assert_array_equal(result, [4, 6])


def test_transform_embedding_supports_bundle_pca_before_scaler():
    wrapper = QKNNWrapper()
    wrapper.pca_pipeline = {
        "pca": _PCA(n_features_in=3, components=2),
        "scaler": _Scaler(n_features_in=2, offset=5),
    }

    result = wrapper._transform_embedding(np.array([1, 2, 3], dtype=np.float32))

    np.testing.assert_array_equal(result, [7, 9])


def test_predict_embedding_uses_bundle_k_and_returns_prediction_contract(monkeypatch):
    _fake_pennylane(monkeypatch)
    wrapper = QKNNWrapper()
    wrapper.qknn_model = {"k": 1}
    wrapper.pca_pipeline = {"pca": None, "scaler": None}
    wrapper.prototypes = {
        "X": np.array([[0, 0], [0.1, 0.1], [0.2, 0.2]], dtype=np.float32),
        "y": np.array([2, 1, 1]),
    }

    result = wrapper.predict_embedding(np.array([0, 0], dtype=np.float32))

    assert result["class_name"] == "Stone"
    assert result["predicted_class"] == "Stone"
    assert result["probabilities"]["Stone"] == 1.0
    assert list(result["probabilities"]) == ["Cyst", "Normal", "Stone", "Tumor"]
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert set(result) == {
        "model_id",
        "model_name",
        "model_family",
        "class_name",
        "predicted_class",
        "confidence",
        "probabilities",
        "inference_time_sec",
    }


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_predict_embedding_rejects_invalid_k(monkeypatch, k):
    _fake_pennylane(monkeypatch)
    wrapper = QKNNWrapper()
    wrapper.qknn_model = {}
    wrapper.pca_pipeline = {"pca": None, "scaler": None}
    wrapper.prototypes = {"X": np.array([[0, 0]], dtype=np.float32), "y": np.array([0])}

    with pytest.raises(ValueError, match="positive integer"):
        wrapper.predict_embedding(np.array([0, 0], dtype=np.float32), k=k)


def test_load_accepts_current_qknn_bundle(monkeypatch, tmp_path):
    _fake_pennylane(monkeypatch)
    bundle_path = tmp_path / "qknn_bundle.joblib"
    bundle_path.touch()
    bundle = {
        "pca": _PCA(n_features_in=3, components=2),
        "scaler": _Scaler(n_features_in=2, offset=0),
        "proto_X": np.zeros((2, 2), dtype=np.float32),
        "proto_y": np.array([0, 1]),
        "k": 1,
    }
    monkeypatch.setattr("dl.models.wrappers.qknn_wrapper.joblib.load", lambda _path: bundle)
    wrapper = QKNNWrapper(artifact_dir=tmp_path, dinov3_wrapper=SimpleNamespace(is_ready=True))

    assert wrapper.load() is True
    assert wrapper.qknn_model["k"] == 1
    np.testing.assert_array_equal(wrapper.prototypes["y"], [0, 1])


def test_load_accepts_legacy_artifact_trio(monkeypatch, tmp_path):
    _fake_pennylane(monkeypatch)
    for name in ("qknn_model.pkl", "pca_pipeline.pkl"):
        (tmp_path / name).touch()
    np.savez(
        tmp_path / "prototypes.npz",
        X=np.zeros((4, 2), dtype=np.float32),
        y=np.arange(4, dtype=np.int64),
    )
    model = {"classes": ["Cyst", "Normal", "Stone", "Tumor"]}
    pipeline = {"pca": _PCA(n_features_in=3, components=2), "scaler": _Scaler(3, 0)}
    monkeypatch.setattr(
        "dl.models.wrappers.qknn_wrapper.joblib.load",
        lambda path: model if path.name == "qknn_model.pkl" else pipeline,
    )
    wrapper = QKNNWrapper(artifact_dir=tmp_path, dinov3_wrapper=SimpleNamespace(is_ready=True))

    assert wrapper.load() is True
    assert wrapper.qknn_model is model
    assert wrapper.prototypes["y"].tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("features", "labels"),
    [
        (np.zeros((2, 2), dtype=np.float32), np.array([0, 4])),
        (np.array([[0, np.nan]], dtype=np.float32), np.array([0])),
        (np.zeros((2, 2), dtype=np.float32), np.array([0.0, 1.0])),
    ],
)
def test_load_rejects_invalid_prototype_values(monkeypatch, tmp_path, features, labels):
    _fake_pennylane(monkeypatch)
    bundle_path = tmp_path / "qknn_bundle.joblib"
    bundle_path.touch()
    bundle = {
        "pca": None,
        "scaler": None,
        "proto_X": features,
        "proto_y": labels,
    }
    monkeypatch.setattr("dl.models.wrappers.qknn_wrapper.joblib.load", lambda _path: bundle)
    wrapper = QKNNWrapper(artifact_dir=tmp_path, dinov3_wrapper=SimpleNamespace(is_ready=True))

    assert wrapper.load() is False
    assert wrapper.status.value == "PENDING_WEIGHTS"
    assert "prototypes" in wrapper.error_message.lower()
