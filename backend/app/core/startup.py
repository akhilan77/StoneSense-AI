"""Startup configuration logic for loading ML and DL model singletons."""

import logging
from app.services.model_loader import model_loader

logger = logging.getLogger("AppStartup")


def load_models_on_startup() -> None:
    """Pre-loads ResNet18 and XGBoost model singletons to RAM."""
    try:
        model_loader.load_all_models()
        logger.info("All model singletons successfully initialized on startup.")
    except Exception as e:
        logger.exception(f"Critical error occurred during model pre-loading: {e}")
