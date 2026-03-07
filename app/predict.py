"""Business logic for image prediction."""

from __future__ import annotations

import io
import logging

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms

from app.model_loader import get_class_names
from app.schemas import PredictionResponse

logger = logging.getLogger(__name__)


def process_image(image_bytes: bytes, transform: transforms.Compose) -> torch.Tensor:
    """Read bytes, convert to PIL Image, and apply transforms.

    Args:
        image_bytes: The raw bytes uploaded by the user.
        transform: The torchvision transform pipeline.

    Returns:
        A torch.Tensor ready to be fed into the model with batch dimension added.

    Raises:
        ValueError: If the image cannot be opened or is corrupted.
    """
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))

        # Ensure image is in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply transforms and add batch dimension using unsqueeze
        tensor = transform(image)
        return tensor.unsqueeze(0)

    except Exception as e:
        logger.exception("Failed to process uploaded image.")
        raise ValueError("Invalid image file.") from e


def predict_image(
    tensor: torch.Tensor, model: nn.Module, device: str = "cpu"
) -> PredictionResponse:
    """Run the image tensor through the ResNet model and return the prediction.

    Args:
        tensor: The preprocessed image tensor of shape (1, C, H, W).
        model: The trained ResNet-152 model.
        device: The device the model is running on ("cpu" or "cuda").

    Returns:
        A PredictionResponse object containing the top predicted class and confidence.
    """
    logger.debug("Running tensor through model.")
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)
        # Apply Softmax to get probabilities
        probabilities = F.softmax(outputs, dim=1)

        # Get the max probability and its index
        top_prob, top_idx = torch.max(probabilities, 1)

    predicted_label_idx = top_idx.item()
    confidence = top_prob.item()

    # Map index to class name
    class_names = get_class_names()
    prediction = class_names[predicted_label_idx]

    logger.info(f"Prediction successful: {prediction} ({confidence:.2f})")

    return PredictionResponse(
        prediction=prediction,
        confidence=confidence,
    )
