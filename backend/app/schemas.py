from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    title: str = Field(default="", description="News headline (optional but improves accuracy)")
    text: str = Field(..., min_length=10, description="Full article body text")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Scientists announce breakthrough in renewable energy storage",
                "text": (
                    "Researchers at a leading university published a peer-reviewed study "
                    "describing a new battery technology that could significantly improve "
                    "energy storage efficiency, according to the report released Monday."
                ),
            }
        }


class PredictResponse(BaseModel):
    label: str            # "Fake" or "Real"
    is_fake: bool
    confidence: float      # 0-1 probability of the predicted class
    fake_probability: float
    real_probability: float
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
