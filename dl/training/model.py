"""ResNet18 Model Architecture Definition for CT Kidney Classification.

Defines transfer learning ResNet18 classifier for 4-class CT kidney scans
(Normal, Cyst, Stone, Tumor). Supports backbone freezing and selective fine-tuning.
"""

from typing import Dict
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# Explicit class ordering required
CLASS_MAPPING: Dict[int, str] = {
    0: "Cyst",
    1: "Normal",
    2: "Stone",
    3: "Tumor"
}


def build_resnet18_classifier(
    num_classes: int = 4,
    freeze_backbone: bool = True,
    unfreeze_layer4: bool = True
) -> nn.Module:
    """Builds ResNet18 model initialized with ImageNet weights.
    
    Args:
        num_classes: Number of output classes (default=4).
        freeze_backbone: If True, freezes feature extractor parameters.
        unfreeze_layer4: If True, unfreezes layer4 for fine-tuning.
        
    Returns:
        torch.nn.Module: Configured ResNet18 classifier.
    """
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    if freeze_backbone and unfreeze_layer4:
        # Unfreeze final residual block (layer4) for domain adaptation
        for param in model.layer4.parameters():
            param.requires_grad = True

    # Replace final fully connected classification layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
