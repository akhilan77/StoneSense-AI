"""Image utilities helper functions for CT Scan preprocessing and security validation."""

import io
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException, UploadFile
from PIL import Image, ImageStat

from app.config.settings import settings

logger = logging.getLogger("ImageUtils")

CHUNK_SIZE = 64 * 1024  # 64 KB read buffer


async def read_bounded_image_bytes(
    upload_file: UploadFile,
    max_bytes: Optional[int] = None,
) -> bytes:
    """Reads uploaded file stream in bounded chunks up to max_bytes to prevent memory exhaustion.

    Args:
        upload_file: FastAPI UploadFile object.
        max_bytes: Maximum allowed payload in bytes. Defaults to settings.ct_upload_max_bytes.

    Returns:
        bytes: Raw image file bytes.

    Raises:
        HTTPException(413): If upload payload exceeds max_bytes.
        HTTPException(422): If upload payload is 0 bytes.
    """
    effective_max = max_bytes if max_bytes is not None else settings.ct_upload_max_bytes
    buffer = bytearray()

    while True:
        chunk = await upload_file.read(CHUNK_SIZE)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > effective_max:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File exceeds maximum allowed upload size of {effective_max} bytes "
                    f"({effective_max / (1024 * 1024):.1f} MB)."
                ),
            )

    if len(buffer) == 0:
        raise HTTPException(
            status_code=422,
            detail="Empty file upload: image file contains 0 bytes.",
        )

    return bytes(buffer)


def validate_ct_image_content(
    image_bytes: bytes,
) -> Tuple[Image.Image, List[str], Dict[str, float]]:
    """Validates CT image by true content: magic bytes, PIL parser, decompression limits, and dimensions.

    Allowed formats: PNG and JPEG only.

    Args:
        image_bytes: Raw binary uploaded image bytes.

    Returns:
        Tuple: (PIL Image object, list of warnings, sanity metrics dict).

    Raises:
        HTTPException(415): Unsupported file format (verified by magic bytes).
        HTTPException(413): Decompression bomb detected (exceeds max safe pixels).
        HTTPException(422): Corrupted image, degenerate size, extreme aspect ratio, or blank/constant.
    """
    if len(image_bytes) == 0:
        raise HTTPException(status_code=422, detail="Empty upload: image contains 0 bytes.")

    # 1. Magic bytes verification (independent of client-supplied Content-Type or file extension)
    is_jpeg = image_bytes.startswith(b"\xff\xd8\xff")
    is_png = image_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    if not (is_jpeg or is_png):
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported media type. Only PNG and JPEG format CT scans are accepted "
                "(verified by file content signature)."
            ),
        )

    # 2. Decompression bomb protection
    Image.MAX_IMAGE_PIXELS = settings.ct_upload_max_pixels

    try:
        # Initial parse & integrity verification
        verify_img = Image.open(io.BytesIO(image_bytes))
        verify_img.verify()
    except Image.DecompressionBombError:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Image decompression bomb detected: pixel count exceeds safe threshold "
                f"of {settings.ct_upload_max_pixels} pixels."
            ),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Corrupted or unreadable image file: {str(exc)}",
        )

    # 3. Reload image data (PIL verify closes the file pointer)
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Image.DecompressionBombError:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Image decompression bomb detected: pixel count exceeds safe threshold "
                f"of {settings.ct_upload_max_pixels} pixels."
            ),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to load image pixels after verification: {str(exc)}",
        )

    width, height = img.size

    # 4. Degenerate dimension checks
    min_dim = settings.ct_upload_min_dimension
    if width < min_dim or height < min_dim:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Image dimensions ({width}x{height}) are too small. "
                f"Minimum required dimension is {min_dim}x{min_dim} pixels."
            ),
        )

    # 5. Aspect ratio check
    aspect_ratio = max(width / height, height / width)
    if aspect_ratio > settings.ct_ood_max_aspect_ratio:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Extreme aspect ratio ({aspect_ratio:.2f}:1 exceeds limit {settings.ct_ood_max_aspect_ratio}:1). "
                f"CT scan slices must be approximately square axial sections."
            ),
        )

    # 6. Near-constant image check (blank or monochromatic upload)
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    stddev = stat.stddev[0] if stat.stddev else 0.0

    warnings: List[str] = []
    if stddev < settings.ct_ood_min_std:
        warnings.append(
            f"Near-constant image detected (pixel std dev {stddev:.2f} < {settings.ct_ood_min_std}). "
            f"Image appears uniform or blank and may lack clinical diagnostic detail."
        )

    # 7. Chromatic sanity checks (medical CT scans are grayscale)
    if img.mode in ("RGB", "RGBA"):
        rgb_img = img.convert("RGB")
        r, g, b = rgb_img.split()
        r_stat = ImageStat.Stat(r).mean[0]
        g_stat = ImageStat.Stat(g).mean[0]
        b_stat = ImageStat.Stat(b).mean[0]
        channel_delta = max(abs(r_stat - g_stat), abs(r_stat - b_stat), abs(g_stat - b_stat))
        if channel_delta > 15.0:
            warnings.append(
                "High chromatic variance detected: medical CT scans are typically grayscale. "
                "Input may be a color photograph or synthetic image."
            )

    sanity_metrics = {
        "width": float(width),
        "height": float(height),
        "aspect_ratio": round(aspect_ratio, 2),
        "pixel_std": round(stddev, 2),
    }

    return img, warnings, sanity_metrics


def evaluate_ct_ood_gate(
    probabilities: Dict[str, float],
    confidence: float,
    warnings: List[str],
) -> Tuple[str, Optional[str], Dict[str, float]]:
    """Evaluates an out-of-distribution (OOD) / low-confidence safety gate on model outputs.

    Note: A softmax-threshold and entropy gate is a heuristic that flags high uncertainty
    and obvious non-conforming inputs. It reduces, but does NOT completely eliminate,
    out-of-distribution or non-kidney CT inputs.

    Args:
        probabilities: Softmax class probability distribution dictionary.
        confidence: Maximum softmax probability.
        warnings: List of sanity warnings from image content validation.

    Returns:
        Tuple: (status, review_reason, ood_metrics).
               status is 'accepted' or 'needs_review'.
    """
    # Calculate Shannon predictive entropy (in nats)
    entropy = -sum(
        p * math.log(p)
        for p in probabilities.values()
        if p > 1e-9
    )

    is_low_conf = confidence < settings.ct_ood_min_confidence
    is_high_entropy = entropy > settings.ct_ood_max_entropy

    ood_metrics = {
        "confidence": round(float(confidence), 4),
        "entropy": round(float(entropy), 4),
        "min_confidence_threshold": float(settings.ct_ood_min_confidence),
        "max_entropy_threshold": float(settings.ct_ood_max_entropy),
    }

    if is_low_conf or is_high_entropy:
        reasons = []
        if is_low_conf:
            reasons.append(
                f"top-class confidence {confidence:.2f} is below safety gate {settings.ct_ood_min_confidence:.2f}"
            )
        if is_high_entropy:
            reasons.append(
                f"predictive entropy {entropy:.2f} nats exceeds threshold {settings.ct_ood_max_entropy:.2f} nats"
            )
        reason_msg = (
            f"Review recommended: {'; '.join(reasons)}. "
            f"Input may be out-of-distribution, degraded, or clinically ambiguous."
        )
        return "needs_review", reason_msg, ood_metrics

    return "accepted", None, ood_metrics


def preprocess_ct_image(image_bytes: bytes, transform: Any) -> Any:
    """Preprocesses raw uploaded CT scan image bytes into PyTorch tensor.

    Args:
        image_bytes: Raw binary uploaded image bytes.
        transform: Compose transform pipeline.

    Returns:
        torch.Tensor: Preprocessed image tensor shaped [1, 3, 224, 224].
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(img).unsqueeze(0)
