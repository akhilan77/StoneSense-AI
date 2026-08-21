"""Image utilities helper functions for CT Scan preprocessing."""

import logging
from typing import Any
from PIL import Image

logger = logging.getLogger("ImageUtils")


def preprocess_ct_image(image_bytes: bytes, transform: Any) -> Any:
    """Preprocesses raw uploaded CT scan image bytes into PyTorch tensor.
    
    Args:
        image_bytes: Raw binary uploaded image bytes.
        transform: Compose transform pipeline.
        
    Returns:
        torch.Tensor: Preprocessed image tensor shaped [1, 3, 224, 224].
    """
    import io
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(img).unsqueeze(0)
