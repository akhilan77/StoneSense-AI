"""ResNet18 Model Wrapper for CT Kidney Classification.

Encapsulates PyTorch ResNet18 loading, forward pass, Softmax probabilities,
and convolutional Grad-CAM explainability.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import io
import time
import logging
import numpy as np
from PIL import Image

from dl.models.base import BaseCTModel, DLModelStatus, CLASS_MAPPING, CLASS_NAMES

logger = logging.getLogger("ResNet18Wrapper")

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class ResNet18Wrapper(BaseCTModel):
    """Wrapper for trained PyTorch ResNet18 CT image classifier."""

    def __init__(
        self,
        artifact_dir: Optional[Path] = None,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[Any] = None,
    ):
        super().__init__(
            model_id="resnet18",
            model_name="ResNet18",
            model_family="resnet18_ct",
            artifact_dir=artifact_dir or (PROJECT_ROOT / "dl" / "models" / "ct" / "resnet18"),
        )
        self.weights_path = Path(weights_path) if weights_path else None
        self._device = device
        self.model = None
        self.transform = None

    def load(self) -> bool:
        """Loads ResNet18 PyTorch state dict and validation transforms."""
        try:
            import torch
            from dl.training.model import build_resnet18_classifier
            from dl.preprocessing.transforms import get_val_test_transforms

            if self._device is None:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Determine weights path with fallbacks
            candidates = []
            if self.weights_path:
                candidates.append(self.weights_path)
            if self.artifact_dir:
                candidates.append(self.artifact_dir / "kidney_resnet18.pth")
                candidates.append(self.artifact_dir / "best_checkpoint.pth")
            candidates.extend([
                PROJECT_ROOT / "dl" / "models" / "kidney_resnet18.pth",
                PROJECT_ROOT / "dl" / "models" / "best_checkpoint.pth",
            ])

            target_weights = None
            for p in candidates:
                if p and p.exists():
                    target_weights = p
                    break

            if not target_weights:
                self._status = DLModelStatus.PENDING_WEIGHTS
                self._error_message = "No ResNet18 weights file found (.pth)"
                logger.warning(self._error_message)
                return False

            self.model = build_resnet18_classifier(num_classes=4, freeze_backbone=False)
            checkpoint = torch.load(target_weights, map_location=self._device, weights_only=False)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.to(self._device)
            self.model.eval()

            self.transform = get_val_test_transforms(image_size=(224, 224))

            # Load metrics from artifact or central dl/models
            metrics_loaded = self.load_metrics_artifact()
            if not metrics_loaded:
                central_metrics = PROJECT_ROOT / "dl" / "models" / "metrics.json"
                if central_metrics.exists():
                    self.load_metrics_artifact(central_metrics)

            self.load_model_card_artifact()

            self._status = DLModelStatus.READY
            self._error_message = None
            logger.info(f"ResNet18Wrapper loaded successfully from {target_weights} on {self._device}")
            return True

        except Exception as exc:
            self._status = DLModelStatus.ERROR
            self._error_message = f"Failed to load ResNet18: {exc}"
            logger.exception(self._error_message)
            return False

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """Runs forward pass on image bytes using loaded ResNet18 model."""
        if not self.is_ready or self.model is None:
            raise RuntimeError("ResNet18 is not ready or loaded for inference.")

        import torch

        start_t = time.perf_counter()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor_img = self.transform(img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            outputs = self.model(tensor_img)
            probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
            pred_idx = int(np.argmax(probs))

        elapsed = time.perf_counter() - start_t
        predicted_class = CLASS_MAPPING.get(pred_idx, "Unknown")
        confidence = float(probs[pred_idx])
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

    def explain(
        self,
        image_bytes: bytes,
        target_class: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generates class-activation map (Grad-CAM) from layer4."""
        if not self.is_ready or self.model is None:
            return super().explain(image_bytes, target_class, output_path)

        try:
            import cv2
            import torch

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            original_rgb = np.asarray(image, dtype=np.uint8)
            tensor = self.transform(image).unsqueeze(0).to(self._device)
            tensor.requires_grad_(True)

            activations = []
            gradients = []
            target_layer = getattr(self.model.layer4[-1], "conv2", self.model.layer4[-1])

            def _fwd_hook(m, inp, out):
                activations.append(out.detach())

            def _bwd_hook(m, gin, gout):
                gradients.append(gout[0].detach())

            h_fwd = target_layer.register_forward_hook(_fwd_hook)
            h_bwd = target_layer.register_full_backward_hook(_bwd_hook)

            try:
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)
                pred_idx = int(torch.argmax(logits, dim=1).item())
                pred_label = CLASS_MAPPING.get(pred_idx, "Normal")

                target_idx = pred_idx
                if target_class:
                    for idx, label in CLASS_MAPPING.items():
                        if label.lower() == str(target_class).lower():
                            target_idx = idx
                            break

                self.model.zero_grad(set_to_none=True)
                score = logits[:, target_idx].sum()
                score.backward()
            finally:
                h_fwd.remove()
                h_bwd.remove()

            if not activations or not gradients:
                return {
                    "available": False,
                    "overlay_url": "",
                    "target_class": CLASS_MAPPING.get(target_idx, pred_label),
                    "message": "Grad-CAM hooks captured no activations."
                }

            act = activations[-1]
            grad = gradients[-1]
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * act).sum(dim=1, keepdim=True))[0, 0].detach().cpu().numpy()

            cam_min, cam_max = float(cam.min()), float(cam.max())
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
            else:
                cam = np.zeros_like(cam)

            heatmap_resized = cv2.resize(cam, (original_rgb.shape[1], original_rgb.shape[0]))
            heatmap = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            orig_float = original_rgb.astype(np.float32) / 255.0
            overlay = np.clip(0.5 * orig_float + 0.5 * heatmap_rgb, 0.0, 1.0)
            overlay_u8 = (overlay * 255).astype(np.uint8)

            saved_path = None
            if output_path:
                out_p = Path(output_path)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_p), cv2.cvtColor(overlay_u8, cv2.COLOR_RGB2BGR))
                saved_path = str(out_p)

            is_normal = (pred_label == "Normal" and not target_class) or (target_class == "Normal")
            return {
                "available": not is_normal,
                "overlay_path": saved_path,
                "target_class": CLASS_MAPPING.get(target_idx, pred_label),
                "prediction": pred_label,
                "confidence": round(float(probs[0, target_idx].item()), 4),
                "message": (
                    "No stone-specific localization is shown because the model classified this scan as Normal."
                    if is_normal
                    else ""
                )
            }
        except Exception as e:
            logger.warning(f"Grad-CAM generation failed: {e}")
            return {
                "available": False,
                "overlay_url": "",
                "target_class": target_class or "Unknown",
                "message": f"Explanation failed: {e}"
            }
