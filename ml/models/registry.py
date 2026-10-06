"""Model Registry for Tabular Kidney Stone Risk Models.

Manages model versions, metadata manifests, artifact persistence, and active deployment pointers.
"""

from pathlib import Path
import json
import logging
from typing import Dict, List, Any, Optional, Union
import joblib

from .base import BaseRiskModel
from .wrappers import (
    LogisticRegressionRiskModel,
    RandomForestRiskModel,
    XGBoostRiskModel,
    FAMILY_MODEL_MAP
)

logger = logging.getLogger("ModelRegistry")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_VERSIONS_DIR = BASE_DIR / "versions"
DEFAULT_MANIFEST_PATH = BASE_DIR / "registry_manifest.json"


class ModelRegistry:
    """Central registry responsible for model versioning, loading, and deployment state."""

    def __init__(
        self,
        versions_dir: Optional[Union[str, Path]] = None,
        manifest_path: Optional[Union[str, Path]] = None
    ):
        self.versions_dir = Path(versions_dir or DEFAULT_VERSIONS_DIR)
        self.manifest_path = Path(manifest_path or DEFAULT_MANIFEST_PATH)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        self._manifest: Dict[str, Any] = {
            "active_model_version": None,
            "models": {}
        }
        self._load_manifest()

    def _load_manifest(self) -> None:
        """Loads or initializes the registry manifest file."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self._manifest = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load registry manifest from {self.manifest_path}: {e}")
        else:
            self._save_manifest()

    def _save_manifest(self) -> None:
        """Persists the registry manifest to disk."""
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=4)

    def register_model(
        self,
        model: BaseRiskModel,
        metadata: Optional[Dict[str, Any]] = None,
        set_active: bool = False
    ) -> str:
        """Registers a BaseRiskModel instance into the registry.
        
        Args:
            model: Instance of BaseRiskModel.
            metadata: Evaluation metrics, training configuration, threshold, etc.
            set_active: Whether to mark this model as the currently active/deployed model.
            
        Returns:
            version_tag: Registered version identifier.
        """
        if not isinstance(model, BaseRiskModel):
            raise TypeError(f"Model must inherit from BaseRiskModel, got {type(model)}")

        if model.model_family not in FAMILY_MODEL_MAP:
            raise ValueError(
                f"Unsupported model family '{model.model_family}'. "
                f"Supported families: {list(FAMILY_MODEL_MAP.keys())}"
            )

        version_tag = model.version_tag
        artifact_filename = f"{version_tag}.pkl"
        artifact_path = self.versions_dir / artifact_filename

        # Save model artifact
        model.save(artifact_path)

        try:
            rel_artifact_path = str(artifact_path.relative_to(BASE_DIR.parent.parent)).replace("\\", "/")
        except ValueError:
            rel_artifact_path = str(artifact_path).replace("\\", "/")

        meta = dict(metadata or model.metadata or {})
        model_entry = {
            "model_family": model.model_family,
            "version_tag": version_tag,
            "artifact_path": rel_artifact_path,
            "absolute_artifact_path": str(artifact_path.resolve()).replace("\\", "/") if artifact_path.is_absolute() else str(artifact_path).replace("\\", "/"),
            "features": model.feature_names,
            "threshold": model.threshold,
            "metrics": meta.get("metrics", {}),
            "trained_at": meta.get("trained_at"),
            "is_deployed": set_active or (self._manifest.get("active_model_version") == version_tag)
        }


        self._manifest["models"][version_tag] = model_entry

        if set_active or self._manifest.get("active_model_version") is None:
            self.set_active_model(version_tag)
        else:
            self._save_manifest()

        logger.info(f"Registered model {version_tag} ({model.model_family}) to {artifact_path}")
        return version_tag

    def set_active_model(self, version_tag: str) -> None:
        """Sets the designated version as the active deployed model."""
        if version_tag not in self._manifest["models"]:
            raise KeyError(f"Model version '{version_tag}' is not registered.")

        for v_tag, entry in self._manifest["models"].items():
            entry["is_deployed"] = (v_tag == version_tag)

        self._manifest["active_model_version"] = version_tag
        self._save_manifest()
        logger.info(f"Active model set to '{version_tag}'")

    def get_model(self, version_tag: Optional[str] = None) -> BaseRiskModel:
        """Loads and returns a model instance by version tag (or active if None)."""
        if version_tag is None:
            version_tag = self._manifest.get("active_model_version")
            if not version_tag:
                raise RuntimeError("No active model is currently configured in the registry.")

        if version_tag not in self._manifest["models"]:
            raise KeyError(f"Model version '{version_tag}' not found in registry manifest.")

        entry = self._manifest["models"][version_tag]
        family = entry["model_family"]
        artifact_path = Path(entry.get("absolute_artifact_path") or (self.versions_dir / f"{version_tag}.pkl"))

        if not artifact_path.exists():
            # Try relative path fallback
            fallback_path = BASE_DIR.parent.parent / entry.get("artifact_path", "")
            if fallback_path.exists():
                artifact_path = fallback_path
            else:
                raise FileNotFoundError(f"Model artifact file not found at {artifact_path}")

        wrapper_cls = FAMILY_MODEL_MAP.get(family)
        if wrapper_cls is None:
            raise ValueError(f"Unknown model family: {family}")

        model = wrapper_cls.load(artifact_path)
        model.version_tag = version_tag
        model.model_family = family
        model.threshold = float(entry.get("threshold", 0.50))
        model.feature_names = entry.get("features", model.feature_names)
        model.metadata = entry
        return model

    def get_model_by_family(self, model_family: str, version_tag: Optional[str] = None) -> BaseRiskModel:
        """Resolves and returns a model belonging to a specific model family."""
        if model_family not in FAMILY_MODEL_MAP:
            raise ValueError(
                f"Invalid model family '{model_family}'. "
                f"Valid families: {list(FAMILY_MODEL_MAP.keys())}"
            )

        if version_tag is not None:
            model = self.get_model(version_tag)
            if model.model_family != model_family:
                raise ValueError(
                    f"Requested version '{version_tag}' has family '{model.model_family}', "
                    f"expected '{model_family}'."
                )
            return model

        # If version not specified, find latest/active model with this family
        active_ver = self._manifest.get("active_model_version")
        if active_ver and self._manifest["models"].get(active_ver, {}).get("model_family") == model_family:
            return self.get_model(active_ver)

        # Look for any registered model with this family
        for v_tag, entry in reversed(list(self._manifest["models"].items())):
            if entry.get("model_family") == model_family:
                return self.get_model(v_tag)

        raise KeyError(f"No model found for family '{model_family}' in registry.")

    def get_active_model(self) -> BaseRiskModel:
        """Returns the currently active deployed BaseRiskModel instance."""
        return self.get_model()

    def get_active_metadata(self) -> Dict[str, Any]:
        """Returns metadata dictionary for the active deployed model."""
        active_ver = self._manifest.get("active_model_version")
        if not active_ver or active_ver not in self._manifest["models"]:
            return {}
        return dict(self._manifest["models"][active_ver])

    def list_models(self) -> List[Dict[str, Any]]:
        """Returns a list of all registered models and their metadata."""
        return list(self._manifest["models"].values())


# Default global registry singleton
registry = ModelRegistry()
