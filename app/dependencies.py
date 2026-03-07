from __future__ import annotations

import logging
import pathlib
from functools import lru_cache

import torch
from torch import nn
from torchvision import transforms

from app.model_loader import load_model

logger = logging.getLogger(__name__)

# Assuming artifacts directory is at the project root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "best_model.pth"


@lru_cache(maxsize=1)
def get_device() -> str:
    """Determine the best available device.

    Returns:
        "cuda" if available, else "cpu".
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_model() -> nn.Module:
    """Load and cache the ResNet-152 model.

    Returns:
        The loaded PyTorch model in evaluation mode.
    """
    device = get_device()
    logger.info("Initializing cached model dependency.")
    try:
        model = load_model(model_path=DEFAULT_MODEL_PATH, device=device)
        return model
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        # In a real scenario, we might want to gracefully degrade or halt,
        # but here we'll raise so the API fails loudly if the model is missing.
        raise


@lru_cache(maxsize=1)
def get_transforms() -> transforms.Compose:
    """Get the image preprocessing transforms.

    These must exactly match the transforms used during training inside
    `src.data.loader.get_data_loaders`.

    Returns:
        A composed torchvision transform.
    """
    logger.debug("Initializing cached transforms dependency.")
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
