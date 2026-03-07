from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictionResponse(BaseModel):
    """Response model for lesion classification predictions."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": "mel",
                "confidence": 0.95,
            }
        }
    )

    prediction: str = Field(
        ...,
        description="The predicted class name of the skin lesion (e.g., 'mel', 'nv').",
    )
    confidence: float = Field(
        ...,
        description="The confidence probability of the prediction, ranging from 0.0 to 1.0.",
        ge=0.0,
        le=1.0,
    )
