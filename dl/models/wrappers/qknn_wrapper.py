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

    def load(self) -> bool:
        """Attempts to load QKNN model, PCA scaler, and quantum circuit prototypes."""
        self.load_metrics_artifact()
        self.load_model_card_artifact()

        if not self.artifact_dir or not self.artifact_dir.exists():
            self._status = DLModelStatus.PENDING_WEIGHTS
            self._error_message = "QKNN artifact directory not found. Waiting for training export."
            return False

        model_path = self.artifact_dir / "qknn_model.pkl"
        pca_path = self.artifact_dir / "pca_pipeline.pkl"
        proto_path = self.artifact_dir / "prototypes.npz"

        if not (model_path.exists() and pca_path.exists() and proto_path.exists()):
            self._status = DLModelStatus.PENDING_WEIGHTS
            self._error_message = "QKNN artifacts (qknn_model.pkl, pca_pipeline.pkl, prototypes.npz) are pending."
            return False

        try:
            self.qknn_model = joblib.load(model_path)
            self.pca_pipeline = joblib.load(pca_path)
            self.prototypes = np.load(proto_path)
        except Exception as exc:
            self._status = DLModelStatus.PENDING_WEIGHTS
            self._error_message = f"Failed to load QKNN components: {exc}"
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

    def predict_embedding(self, embedding: np.ndarray, k: int = 40) -> Dict[str, Any]:
        """Runs PennyLane state-fidelity quantum kernel against saved class prototypes on a 384-d vector."""
        if self.pca_pipeline is None or self.prototypes is None:
            if not self.load():
                raise RuntimeError(f"QKNN components not loaded ({self._error_message}).")

        import pennylane as qml

        start_t = time.perf_counter()

        # Extract PCA and Scaler objects
        pca = self.pca_pipeline["pca"] if isinstance(self.pca_pipeline, dict) else getattr(self.pca_pipeline, "pca", self.pca_pipeline)
        scaler = self.pca_pipeline["scaler"] if isinstance(self.pca_pipeline, dict) else getattr(self.pca_pipeline, "scaler", None)

        emb_2d = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if scaler is not None:
            emb_scaled = scaler.transform(emb_2d)
        else:
            emb_scaled = emb_2d

        if pca is not None:
            x_q = pca.transform(emb_scaled)[0]
        else:
            x_q = emb_scaled[0]

        proto_X = self.prototypes["X"]
        proto_y = self.prototypes["y"]
        n_qubits = len(x_q)

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def quantum_fidelity_circuit(x1, x2):
            qml.AngleEmbedding(x1, wires=range(n_qubits), rotation='Y')
            qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits), rotation='Y')
            return qml.probs(wires=range(n_qubits))

        sims = np.array([quantum_fidelity_circuit(x_q, p)[0] for p in proto_X], dtype=np.float32)
        top_k = min(k, len(sims))
        top_k_idx = np.argsort(sims)[-top_k:]
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
