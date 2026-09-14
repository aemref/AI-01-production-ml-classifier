"""HTTP inference interface for the selected baseline model."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from src.predictor import Predictor


DEFAULT_DATA_PATH = (
    Path(__file__).parents[1] / "data" / "breast_cancer_wisconsin_diagnostic.csv"
)


class PredictionRequest(BaseModel):
    """Validated feature payload matching the training data contract."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    feature_a: float = Field(description="Mean radius")
    feature_b: float = Field(description="Mean texture")


class PredictionResponse(BaseModel):
    """Probabilistic binary prediction with an auditable model version."""

    label: int
    label_name: str
    confidence: float
    malignant_probability: float
    benign_probability: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str


def create_app(
    predictor: Predictor | None = None,
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> FastAPI:
    """Create an application, optionally injecting a fitted predictor for tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = predictor or Predictor.from_dataset(data_path)
        yield

    application = FastAPI(
        title="AI-01 Production ML Classifier",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        loaded_predictor: Predictor = request.app.state.predictor
        return HealthResponse(
            status="ok",
            model_version=loaded_predictor.model_version,
        )

    @application.post("/predict", response_model=PredictionResponse)
    def predict(
        payload: PredictionRequest,
        request: Request,
    ) -> PredictionResponse:
        loaded_predictor: Predictor = request.app.state.predictor
        result = loaded_predictor.predict(
            feature_a=payload.feature_a,
            feature_b=payload.feature_b,
        )
        return PredictionResponse(**result.to_dict())

    return application


app = create_app()
