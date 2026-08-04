"""Inference Wrapper Module for CT Kidney Image Classification.

Provides reusable prediction methods for single CT images and image batches
loading trained kidney_resnet18.pth model and class_mapping.json.
"""

from pathlib import Path
import json
import logging
from typing import Dict, List, Any, Union, Optional
from PIL import Image
import torch
import torch.nn as nn

from model import build_resnet18_classifier
from transforms import get_val_test_transforms

logger = logging.getLogger("CTInferenceEngine")


class CTInferenceEngine:
    """Inference engine for CT Kidney Image classification."""

    def __init__(self, model_path: Path, class_mapping_path: Path, device: Optional[torch.device] = None):
        self.model_path = Path(model_path)
        self.class_mapping_path = Path(class_mapping_path)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Load class mapping
        if not self.class_mapping_path.exists():
            raise FileNotFoundError(f"Class mapping not found at {self.class_mapping_path}")
        with open(self.class_mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            self.class_names = [mapping[str(i)] for i in range(len(mapping))]

        # Load model
        self.model = build_resnet18_classifier(num_classes=len(self.class_names), freeze_backbone=False)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        self.transform = get_val_test_transforms(image_size=(224, 224))
        logger.info(f"CTInferenceEngine initialized successfully on device: {self.device}")

    def predict_image(self, image_input: Union[Path, str, Image.Image]) -> Dict[str, Any]:
        """Predicts class and probabilities for a single CT scan image."""
        if isinstance(image_input, (str, Path)):
            with Image.open(image_input) as img:
                img = img.convert("RGB")
                tensor_img = self.transform(img).unsqueeze(0).to(self.device)
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
            tensor_img = self.transform(img).unsqueeze(0).to(self.device)
        else:
            raise TypeError("Unsupported image input type.")

        with torch.no_grad():
            outputs = self.model(tensor_img)
            probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
            pred_idx = int(torch.argmax(outputs, dim=1).item())

        predicted_class = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])

        probabilities = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities
        }
