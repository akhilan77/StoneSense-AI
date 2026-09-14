"""Inference Wrapper Module for CT Kidney Image Classification.

Provides reusable prediction methods for single CT images and image batches
loading trained kidney_resnet18.pth model and class_mapping.json.
"""

import sys
from pathlib import Path
import json
import logging
import argparse
from typing import Dict, List, Any, Union, Optional
from PIL import Image
import torch
import torch.nn as nn

# Ensure sibling dl packages (preprocessing, training) are in python path
current_dir = Path(__file__).resolve().parent
preprocessing_dir = current_dir.parent / "preprocessing"
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
if str(preprocessing_dir) not in sys.path:
    sys.path.insert(0, str(preprocessing_dir))

from model import build_resnet18_classifier
from transforms import get_val_test_transforms

logger = logging.getLogger("CTInferenceEngine")


class CTInferenceEngine:
    """Inference engine for CT Kidney Image classification."""

    def __init__(
        self,
        model_path: Union[str, Path] = "dl/models/kidney_resnet18.pth",
        class_mapping_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists() and (Path("dl/models/kidney_resnet18.pth")).exists():
            self.model_path = Path("dl/models/kidney_resnet18.pth")

        if class_mapping_path is None:
            self.class_mapping_path = self.model_path.parent / "class_mapping.json"
        else:
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
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
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
            "prediction": predicted_class,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities
        }

    predict = predict_image


# Alias for backward and external compatibility
KidneyStoneInferenceEngine = CTInferenceEngine


def main():
    parser = argparse.ArgumentParser(description="CT Kidney Image Classification Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input CT image file")
    parser.add_argument(
        "--weights",
        type=str,
        default="dl/models/kidney_resnet18.pth",
        help="Path to trained model weights (.pth)"
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default="dl/models/class_mapping.json",
        help="Path to class mapping JSON"
    )
    args = parser.parse_args()

    engine = CTInferenceEngine(
        model_path=Path(args.weights),
        class_mapping_path=Path(args.mapping)
    )
    result = engine.predict_image(args.image)
    print("\n--- Inference Result ---")
    print(f"Image: {args.image}")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    print("Class Probabilities:")
    for cls, prob in result["probabilities"].items():
        print(f"  {cls:10s}: {prob * 100:.2f}%")


if __name__ == "__main__":
    main()
