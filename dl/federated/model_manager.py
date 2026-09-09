"""StoneSense-AI Model Version Manager.

Handles saving, versioning, checkpoint registry, and deployment pointers
for centralized and federated ResNet18 checkpoints.
"""

from pathlib import Path
import sys
import shutil
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "dl" / "models"
FED_MODELS_DIR = MODELS_DIR / "federated"

logger = logging.getLogger("StoneSenseModelManager")


class ModelManager:
    """Manages model checkpoints, version tags, and deployment pointers."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.fed_dir = self.models_dir / "federated"
        self.fed_dir.mkdir(parents=True, exist_ok=True)

    def save_round_checkpoint(
        self,
        model_state_dict: Dict[str, torch.Tensor],
        round_number: int,
        metrics: Dict[str, float],
        is_deployed: bool = True
    ) -> Path:
        """Saves a new versioned model checkpoint for a federated round.

        Args:
            model_state_dict: State dict of the aggregated global model.
            round_number: The current FL round index.
            metrics: Aggregated metrics (accuracy, loss, f1_score).
            is_deployed: Whether this round becomes the active deployed model.

        Returns:
            Path to the saved checkpoint.
        """
        version_tag = f"resnet18_fed_round_{round_number:03d}"
        checkpoint_filename = f"{version_tag}.pth"
        checkpoint_path = self.fed_dir / checkpoint_filename

        # Save immutable versioned checkpoint
        torch.save({
            "round_number": round_number,
            "version_tag": version_tag,
            "model_state_dict": model_state_dict,
            "metrics": metrics,
            "saved_at": datetime.utcnow().isoformat(),
        }, checkpoint_path)

        logger.info(f"Versioned checkpoint saved: {checkpoint_path}")

        # Update latest.pth pointer
        latest_path = self.fed_dir / "latest.pth"
        shutil.copy2(checkpoint_path, latest_path)

        # If deployed, update active kidney_resnet18.pth for backend inference
        if is_deployed:
            active_path = self.models_dir / "kidney_resnet18.pth"
            torch.save(model_state_dict, active_path)
            logger.info(f"Active inference weights updated at {active_path}")

        return checkpoint_path

    def get_latest_checkpoint_path(self) -> Optional[Path]:
        """Returns path to the latest available federated checkpoint."""
        latest_path = self.fed_dir / "latest.pth"
        if latest_path.exists():
            return latest_path
        active_path = self.models_dir / "kidney_resnet18.pth"
        return active_path if active_path.exists() else None

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Lists all available federated model checkpoints."""
        checkpoints = []
        for file in sorted(self.fed_dir.glob("resnet18_fed_round_*.pth")):
            try:
                data = torch.load(file, map_location="cpu", weights_only=False)
                checkpoints.append({
                    "version_tag": data.get("version_tag", file.stem),
                    "round_number": data.get("round_number"),
                    "metrics": data.get("metrics", {}),
                    "saved_at": data.get("saved_at"),
                    "file_path": str(file.relative_to(PROJECT_ROOT)),
                })
            except Exception as e:
                logger.warning(f"Could not inspect {file}: {e}")
        return checkpoints


model_manager = ModelManager()
