"""Empirical Evaluation of CT Out-of-Distribution (OOD) and Low-Confidence Safety Gate.

Evaluates ResNet18 max-softmax confidence and Shannon predictive entropy across:
1. In-Distribution (ID): Held-out test slices from dl/processed_grouped/test/
2. Synthetic Out-of-Distribution (OOD): Procedurally generated non-kidney images:
   - Uniform / Gaussian noise
   - Linear and radial gradients
   - Rendered text / document artifacts
   - Constant / solid color images
   - Multi-scale procedural textures (photo-like noise)
"""

import io
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

# Add backend and root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT))

from app.config.settings import settings
from app.utils.image_utils import evaluate_ct_ood_gate, validate_ct_image_content
from dl.models.registry import dl_registry


def load_model():
    """Resolves and returns the deployed ResNet18 model wrapper."""
    from dl.models.base import CLASS_NAMES
    wrapper = dl_registry.get_model("resnet18")
    if not wrapper.is_ready:
        wrapper.load()
    model = wrapper.model
    model.eval()
    return model, CLASS_NAMES


def get_image_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    """Encodes PIL Image into binary bytes."""
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def generate_synthetic_ood_images(count_per_type: int = 25) -> Dict[str, List[Image.Image]]:
    """Procedurally generates diverse non-kidney synthetic images."""
    np.random.seed(42)
    synthetic_sets: Dict[str, List[Image.Image]] = {
        "gaussian_noise": [],
        "linear_gradients": [],
        "text_renders": [],
        "constant_images": [],
        "photo_textures": [],
    }

    # 1. Noise images
    for _ in range(count_per_type):
        noise_arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        synthetic_sets["gaussian_noise"].append(Image.fromarray(noise_arr))

    # 2. Linear and diagonal gradients
    for i in range(count_per_type):
        x = np.linspace(0, 255, 224, dtype=np.uint8)
        grad = np.tile(x, (224, 1))
        if i % 2 == 1:
            grad = grad.T
        grad_rgb = np.stack([grad, np.rot90(grad), 255 - grad], axis=-1)
        synthetic_sets["linear_gradients"].append(Image.fromarray(grad_rgb))

    # 3. Text renders
    for i in range(count_per_type):
        img = Image.new("RGB", (224, 224), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        sample_text = f"Report #{i}\nPatient non-kidney doc\nSample text render {i * 17}\nCONFIDENTIAL"
        draw.text((15, 30), sample_text, fill=(20, 20, 20))
        synthetic_sets["text_renders"].append(img)

    # 4. Constant / near-constant images (various brightness levels)
    for i in range(count_per_type):
        val = int(255 * (i / max(1, count_per_type - 1)))
        synthetic_sets["constant_images"].append(Image.new("RGB", (224, 224), color=(val, val, val)))

    # 5. Multi-scale procedural textures (sine interference patterns)
    for i in range(count_per_type):
        x = np.linspace(0, 10 + i, 224)
        y = np.linspace(0, 10 + i, 224)
        xx, yy = np.meshgrid(x, y)
        pattern = np.sin(xx) * np.cos(yy) + 0.5 * np.sin(2 * xx + yy)
        pattern_norm = ((pattern - pattern.min()) / (pattern.max() - pattern.min()) * 255).astype(np.uint8)
        tex_rgb = np.stack([pattern_norm, pattern_norm, pattern_norm], axis=-1)
        synthetic_sets["photo_textures"].append(Image.fromarray(tex_rgb))

    return synthetic_sets


def evaluate_batch(
    model: torch.nn.Module,
    classes: List[str],
    images: List[Image.Image],
    transform: transforms.Compose,
) -> List[Dict]:
    """Runs inference and gate checks on a list of PIL images."""
    results = []
    for img in images:
        raw_bytes = get_image_bytes(img)
        try:
            _, warnings, _ = validate_ct_image_content(raw_bytes)
        except Exception:
            warnings = ["content_validation_rejected"]

        tensor = transform(img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            probs_t = F.softmax(logits, dim=1).squeeze(0)

        probs_dict = {classes[i]: float(probs_t[i].item()) for i in range(len(classes))}
        conf = float(probs_t.max().item())

        entropy = -sum(p * math.log(p) for p in probs_dict.values() if p > 1e-9)
        gate_status, reason, metrics = evaluate_ct_ood_gate(
            probabilities=probs_dict,
            confidence=conf,
            warnings=warnings,
        )

        results.append({
            "confidence": conf,
            "entropy": entropy,
            "gate_status": gate_status,
            "rejected": gate_status == "needs_review",
            "warnings": warnings,
            "top_class": classes[int(probs_t.argmax().item())],
        })
    return results


def summarize(name: str, results: List[Dict]):
    """Prints formatted summary stats for an evaluation group."""
    confs = [r["confidence"] for r in results]
    entropies = [r["entropy"] for r in results]
    rejections = sum(1 for r in results if r["rejected"])
    total = len(results)
    rate = (rejections / total * 100) if total else 0.0

    print(f"\n--- {name} (N={total}) ---")
    print(f"  Confidence: Mean={np.mean(confs):.4f}, Std={np.std(confs):.4f}, Min={np.min(confs):.4f}, Max={np.max(confs):.4f}")
    print(f"  Entropy:    Mean={np.mean(entropies):.4f}, Std={np.std(entropies):.4f}, Min={np.min(entropies):.4f}, Max={np.max(entropies):.4f} (Max possible ~1.386 nats)")
    print(f"  Gate Status 'needs_review' Rejection Rate: {rejections}/{total} ({rate:.1f}%)")


def main():
    print("=" * 70)
    print("StoneSense-AI: CT Out-of-Distribution (OOD) Gate Empirical Evaluation")
    print("=" * 70)
    print(f"Thresholds applied: min_confidence={settings.ct_ood_min_confidence:.2f}, max_entropy={settings.ct_ood_max_entropy:.2f} nats, min_std={settings.ct_ood_min_std}")

    model, classes = load_model()
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 1. In-distribution evaluation
    test_dir = PROJECT_ROOT / "dl" / "processed_grouped" / "test"
    id_images = []
    if test_dir.exists():
        for cls_name in ["Cyst", "Normal", "Stone", "Tumor"]:
            cls_dir = test_dir / cls_name
            if cls_dir.exists():
                slices = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
                for s in slices[:25]:  # Take 25 per class = 100 slices
                    try:
                        id_images.append(Image.open(s))
                    except Exception:
                        pass

    if id_images:
        id_results = evaluate_batch(model, classes, id_images, eval_transform)
        summarize("IN-DISTRIBUTION: Real Kidney CT Test Slices", id_results)
    else:
        print("Warning: No in-distribution test slices found in dl/processed_grouped/test.")

    # 2. Synthetic out-of-distribution evaluation
    synthetic_sets = generate_synthetic_ood_images(count_per_type=25)
    all_ood_results = []
    for set_name, imgs in synthetic_sets.items():
        ood_res = evaluate_batch(model, classes, imgs, eval_transform)
        summarize(f"SYNTHETIC OOD: {set_name}", ood_res)
        all_ood_results.extend(ood_res)

    summarize("OVERALL SYNTHETIC OOD AGGREGATE", all_ood_results)
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
