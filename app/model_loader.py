from __future__ import annotations

import logging
import pathlib

import torch
import torchvision.models as models
from torch import nn

logger = logging.getLogger(__name__)

# Standard HAM10000 classes based on the dataset structure
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def load_model(
    model_path: str | pathlib.Path, num_classes: int = len(CLASSES), device: str = "cpu"
) -> nn.Module:
    """Load the trained ResNet-152 model.

    Instantiates a ResNet-152 model, modifies the final fully connected
    layer to match the number of diagnosis classes, and loads the weights
    from the provided state dictionary path.

    Args:
        model_path: The filesystem path to the saved PyTorch model (.pth).
        num_classes: The number of output classes.
        device: The device to load the model on ("cpu" or "cuda").

    Returns:
        The instantiated, loaded, and eval-mode ResNet-152 model.

    Raises:
        FileNotFoundError: If the specified `model_path` does not exist.
        RuntimeError: If model weights fail to load correctly.
    """
    path = pathlib.Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found at {path}")

    logger.info("Instantiating ResNet-152 architecture.")
    # Initialize model without pretrained weights since we will load our own
    model = models.resnet152(weights=None)

    # Modify the fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    logger.info(f"Loading weights from {path} to device '{device}'")
    try:
        # Load state dict, map to the chosen device
        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as e:
        logger.exception("Failed to load model weights.")
        raise RuntimeError(f"Failed to load model from {path}: {e}") from e

    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model


def get_class_names() -> list[str]:
    """Return the ordered list of class names.

    Returns:
        A list of string class names.
    """
    return CLASSES
