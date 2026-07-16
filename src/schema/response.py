from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """
    Response schema for customer churn prediction.
    """

    prediction: str = Field(
        ..., description="Predicted churn label", json_schema_extra={"example": "Yes"}
    )

    probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Probability of churn",
        json_schema_extra={"example": 0.91},
    )

    model_version: str = Field(..., json_schema_extra={"example": "1.0.0"})
