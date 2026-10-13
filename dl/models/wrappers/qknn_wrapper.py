"""Quantum K-Nearest Neighbors (QKNN) Model Wrapper for CT Kidney Classification.

Encapsulates PennyLane quantum circuit state-fidelity kernel on top of
PCA-compressed foundation model embeddings.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import io
import time
import logging
import joblib
import numpy as np
from PIL import Image

from dl.models.base import BaseCTModel, DLModelStatus, CLASS_MAPPING, CLASS_NAMES

logger = logging.getLogger("QKNNWrapper")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_K = 40


class QKNNWrapper(BaseCTModel):
    """Wrapper for PennyLane Quantum K-Nearest Neighbors Classifier."""

    def __init__(
        self,
        artifact_dir: Optional[Path] = None,
        dinov3_wrapper: Optional[Any] = None,
    ):
        super().__init__(
            model_id="qknn",
            model_name="QKNN",
            model_family="qknn_ct",
            artifact_dir=artifact_dir or (PROJECT_ROOT / "dl" / "models" / "ct" / "qknn"),
        )
        self.qknn_model = None
        self.pca_pipeline = None
        self.prototypes = None
        self.feature_extractor = None
        self._dinov3_wrapper = dinov3_wrapper
        self.can_federate = False
        self.federation_strategy = None
        self.federation_readiness = "UNSUPPORTED"
        self.federation_ineligibility_reason = "Quantum K-Nearest Neighbors classifier uses fixed quantum circuits and prototype vectors without gradient-trainable weights for FedAvg."

    def load(self) -> bool:
        """Attempts to load QKNN model, PCA scaler, and quantum circuit prototypes."""
        self.load_metrics_artifact()
        self.load_model_card_artifact()

        if not self.artifact_dir or not self.artifact_dir.exists():
            self._status = DLModelStatus.PENDING_WEIGHTS
            self._error_message = "QKNN artifact directory not found. Waiting for training export."
            return False

        bundle_path = self.artifact_dir / "qknn_bundle.joblib"
        model_path = self.artifact_dir / "qknn_model.pkl"
        pca_path = self.artifact_dir / "pca_pipeline.pkl"
        proto_path = self.artifact_dir / "prototypes.npz"

        if bundle_path.exists():
            try:
                bundle = joblib.load(bundle_path)
                required = {"pca", "scaler", "proto_X", "proto_y"}
                if not isinstance(bundle, dict) or not required.issubset(bundle):
                    raise ValueError(
                        f"QKNN bundle is missing required fields: {sorted(required - set(bundle or {}))}."
                    )
                self.qknn_model = bundle
                self.pca_pipeline = {"pca": bundle["pca"], "scaler": bundle["scaler"]}
                self.prototypes = {
                    "X": np.asarray(bundle["proto_X"]),
                    "y": np.asarray(bundle["proto_y"]),
                }
            except Exception as exc:
                self._status = DLModelStatus.PENDING_WEIGHTS
                self._error_message = f"Failed to load QKNN bundle: {exc}"
                return False
        elif model_path.exists() and pca_path.exists() and proto_path.exists():
            try:
                self.qknn_model = joblib.load(model_path)
                self.pca_pipeline = joblib.load(pca_path)
                self.prototypes = np.load(proto_path)
            except Exception as exc:
                self._status = DLModelStatus.PENDING_WEIGHTS
                self._error_message = f"Failed to load QKNN components: {exc}"
                return False
        else:
            self._status = DLModelStatus.PENDING_WEIGHTS
            self._error_message = (
                "QKNN artifacts are pending. Provide qknn_bundle.joblib or the legacy "
                "qknn_model.pkl, pca_pipeline.pkl, and prototypes.npz files."
            )
            return False

        if (
            self.prototypes["X"].ndim != 2
            or self.prototypes["y"].ndim != 1
            or self.prototypes["X"].shape[0] != self.prototypes["y"].shape[0]
            or self.prototypes["X"].shape[0] == 0
            or not np.issubdtype(self.prototypes["X"].dtype, np.number)
            or not np.isfinite(self.prototypes["X"]).all()
            or not np.issubdtype(self.prototypes["y"].dtype, np.integer)
            or np.any(self.prototypes["y"] < 0)
            or np.any(self.prototypes["y"] >= len(CLASS_NAMES))
        ):
            self._status = DLModelStatus.PENDING_WEIGHTS
            self._error_message = (
                "QKNN prototypes must contain finite numeric 2-D features and aligned integer labels "
                f"in [0, {len(CLASS_NAMES) - 1}]."
            )
            return False

        try:
            import pennylane as qml
        except ImportError:
            self._status = DLModelStatus.PENDING_WEIGHTS
            self._error_message = "pennylane is not installed in current environment."
            return False

        # Raw CT image bytes inference requires DINOv3 backbone for feature extraction
        dinov3_backbone_ready = False
        if self._dinov3_wrapper is not None:
            if not self._dinov3_wrapper.is_ready:
                self._dinov3_wrapper.load()
            dinov3_backbone_ready = self._dinov3_wrapper.is_ready
        else:
            try:
                from dl.models.registry import dl_registry
                dino_model = dl_registry.get_model("dinov3")
                if not dino_model.is_ready:
                    dino_model.load()
                dinov3_backbone_ready = dino_model.is_ready
            except Exception:
                dinov3_backbone_ready = False

        if not dinov3_backbone_ready:
            self._status = DLModelStatus.PENDING_WEIGHTS
            self._error_message = "QKNN image prediction requires DINOv3 backbone weights (currently pending)."
            logger.info("QKNNWrapper loaded quantum circuit components; pending DINOv3 backbone for end-to-end images.")
            return False

        self._status = DLModelStatus.READY
        self._error_message = None
        logger.info("QKNNWrapper loaded successfully.")
        return True

    def _transform_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Applies the fitted scaler/PCA in the order encoded by the loaded artifacts."""
        pca = (
            self.pca_pipeline["pca"]
            if isinstance(self.pca_pipeline, dict)
            else getattr(self.pca_pipeline, "pca", self.pca_pipeline)
        )
        scaler = (
            self.pca_pipeline["scaler"]
            if isinstance(self.pca_pipeline, dict)
            else getattr(self.pca_pipeline, "scaler", None)
        )

        emb_2d = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if not np.isfinite(emb_2d).all():
            raise ValueError("QKNN embedding must contain only finite values.")

        if pca is not None and scaler is not None:
            scaler_input_size = getattr(scaler, "n_features_in_", None)
            pca_input_size = getattr(pca, "n_features_in_", None)
            if scaler_input_size == emb_2d.shape[1]:
                # Legacy artifacts fit the scaler on raw DINOv3 embeddings.
                transformed = pca.transform(scaler.transform(emb_2d))
            elif pca_input_size == emb_2d.shape[1]:
                # Current training bundles fit PCA first, then scale the PCA coordinates.
                transformed = scaler.transform(pca.transform(emb_2d))
            else:
                raise ValueError(
                    "QKNN scaler/PCA dimensions do not match the input embedding "
                    f"({emb_2d.shape[1]} features)."
                )
        elif pca is not None:
            transformed = pca.transform(emb_2d)
        elif scaler is not None:
            transformed = scaler.transform(emb_2d)
        else:
            transformed = emb_2d

        return np.asarray(transformed, dtype=np.float32).reshape(-1)

    def predict_embedding(self, embedding: np.ndarray, k: Optional[int] = None) -> Dict[str, Any]:
        """Runs PennyLane state-fidelity quantum kernel against saved class prototypes on a 384-d vector."""
        if self.pca_pipeline is None or self.prototypes is None:
            if not self.load():
                raise RuntimeError(f"QKNN components not loaded ({self._error_message}).")

        import pennylane as qml

        start_t = time.perf_counter()
        x_q = self._transform_embedding(embedding)

        proto_X = self.prototypes["X"]
        proto_y = self.prototypes["y"]
        if proto_X.ndim != 2 or proto_y.ndim != 1 or len(proto_X) != len(proto_y) or len(proto_X) == 0:
            raise ValueError("QKNN prototypes must contain non-empty, aligned 2-D features and 1-D labels.")
        if proto_X.shape[1] != len(x_q):
            raise ValueError(
                f"QKNN transformed embedding has {len(x_q)} features, "
                f"but prototypes have {proto_X.shape[1]}."
            )

        configured_k = self.qknn_model.get("k", DEFAULT_K) if isinstance(self.qknn_model, dict) else DEFAULT_K
        if k is None:
            k = configured_k
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("QKNN k must be a positive integer.")
        k = min(int(k), len(proto_X))
        n_qubits = len(x_q)

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def quantum_fidelity_circuit(x1, x2):
            qml.AngleEmbedding(x1, wires=range(n_qubits), rotation='Y')
            qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits), rotation='Y')
            return qml.probs(wires=range(n_qubits))
        sims = np.array([quantum_fidelity_circuit(x_q, p)[0] for p in proto_X], dtype=np.float32)
        sims = np.array([quantum_fidelity_circuit(x_q, p)[0] for p in proto_X], dtype=np.float32)
        top_k_idx = np.argsort(sims)[-k:]
        top_k_labels = proto_y[top_k_idx]
        top_k_weights = sims[top_k_idx]

        class_scores = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        for lbl, w in zip(top_k_labels, top_k_weights):
            class_scores[int(lbl)] += float(w)

        sum_scores = float(np.sum(class_scores))
        if sum_scores > 0:
            probs = class_scores / sum_scores
        else:
            probs = np.ones(len(CLASS_NAMES), dtype=np.float32) / len(CLASS_NAMES)

        pred_idx = int(np.argmax(probs))
        predicted_class = CLASS_MAPPING.get(pred_idx, "Unknown")
        confidence = float(probs[pred_idx])

        elapsed = time.perf_counter() - start_t
        probabilities = {CLASS_MAPPING[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_family": self.model_family,
            "class_name": predicted_class,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
            "inference_time_sec": round(elapsed, 4),
        }

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """Runs quantum circuit distance query against class prototypes after feature extraction."""
        if not self.is_ready:
            raise RuntimeError(f"QKNN model is not available ({self._error_message}).")

        if self._dinov3_wrapper is not None:
            dino = self._dinov3_wrapper
        else:
            from dl.models.registry import dl_registry
            dino = dl_registry.get_model("dinov3")

        if not dino.is_ready or dino.backbone is None or dino.processor is None:
            raise RuntimeError("QKNN image inference requires DINOv3 backbone feature extractor.")

        import torch
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = dino.processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = dino.backbone(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embedding = outputs.pooler_output.cpu().numpy()[0]
            else:
                embedding = outputs.last_hidden_state[:, 0].cpu().numpy()[0]

        return self.predict_embedding(embedding)

    def explain(
        self,
        image_bytes: bytes,
        target_class: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """QKNN explainability returns prototype alignment note."""
        return {
            "available": False,
            "overlay_url": "",
            "target_class": target_class or "Unknown",
            "message": "Explainability is not supported for Quantum K-Nearest Neighbors."
        }
