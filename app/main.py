"""Main FastAPI application factory and endpoints."""

from __future__ import annotations

import logging
import pathlib

from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from torch import nn
from torchvision import transforms

from app.dependencies import get_model, get_transforms
from app.middleware import setup_middleware
from app.predict import predict_image, process_image
from app.schemas import PredictionResponse

# Configure basic logging for the applicatoin
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        The configured FastAPI instance.
    """
    app = FastAPI(
        title="Skin Lesion Classification API",
        description="A FastAPI prediction service for classifying skin lesions using a ResNet-152 model.",
        version="1.0.0",
    )

    # Attach middlewares and exception handlers
    setup_middleware(app)

    @app.get("/", response_class=HTMLResponse, summary="Home endpoint")
    def home() -> HTMLResponse:
        """Serve the React application UI."""
        index_path = pathlib.Path(__file__).parent / "templates" / "index.html"
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        summary="Predict skin lesion class from an uploaded image.",
        status_code=200,
    )
    async def predict_endpoint(
        file: UploadFile = File(...),
        model: nn.Module = Depends(get_model),
        transform: transforms.Compose = Depends(get_transforms),
    ) -> PredictionResponse:
        """Process an uploaded image and return a classification prediction.

        Args:
            file: The uploaded image file.
            model: The loaded ResNet-152 classification model.
            transform: The torchvision transform pipeline.

        Returns:
            A `PredictionResponse` JSON containing the class and confidence.
        """
        logger.info(f"Received file: {file.filename}")

        # Read the file contents directly into memory
        image_bytes = await file.read()

        # We can perform the preprocessing step synchronously
        tensor = process_image(image_bytes, transform)

        # Run inference logic
        prediction = predict_image(tensor, model)

        return prediction

    return app


# Create the module-level instance required by uvicorn
app = create_app()
