"""Reusable prediction service for the selected baseline classifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from math import exp, isfinite
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.model_artifact import load_artifact
from src.train import FEATURE_COLUMNS, load_and_validate_dataset, split_dataset


DEFAULT_DATA_PATH = Path("data/breast_cancer_wisconsin_diagnostic.csv")
DEFAULT_ARTIFACT_PATH = (
    Path(__file__).parents[1]
    / "artifacts"
    / "logistic-regression-98b12889accb.json"
)
LABEL_NAMES = {0: "malignant", 1: "benign"}


@dataclass(frozen=True)
class Prediction:
    """Serializable result from one model prediction."""

    label: int
    label_name: str
    confidence: float
    malignant_probability: float
    benign_probability: float
    model_version: str

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


class PortableLogisticModel:
    """Minimal inference-only implementation of the persisted model parameters."""

    classes_ = [0, 1]

    def __init__(
        self,
        *,
        scaler_mean: list[float],
        scaler_scale: list[float],
        coefficients: list[float],
        intercept: float,
    ) -> None:
        self._scaler_mean = scaler_mean
        self._scaler_scale = scaler_scale
        self._coefficients = coefficients
        self._intercept = intercept

    @staticmethod
    def _sigmoid(score: float) -> float:
        if score >= 0:
            return 1.0 / (1.0 + exp(-score))
        exponential = exp(score)
        return exponential / (1.0 + exponential)

    def predict_proba(self, features: pd.DataFrame) -> list[list[float]]:
        probabilities = []
        for values in features[FEATURE_COLUMNS].itertuples(index=False, name=None):
            scaled = [
                (float(value) - mean) / scale
                for value, mean, scale in zip(
                    values, self._scaler_mean, self._scaler_scale
                )
            ]
            score = self._intercept + sum(
                coefficient * value
                for coefficient, value in zip(self._coefficients, scaled)
            )
            benign_probability = self._sigmoid(score)
            probabilities.append([1.0 - benign_probability, benign_probability])
        return probabilities


class Predictor:
    """Fit and serve the validation-selected logistic regression baseline."""

    def __init__(self, model: Any, *, model_version: str) -> None:
        self._model = model
        self.model_version = model_version

    @classmethod
    def from_artifact(
        cls, artifact_path: str | Path = DEFAULT_ARTIFACT_PATH
    ) -> "Predictor":
        artifact = load_artifact(artifact_path)
        parameters = artifact["model"]
        model = PortableLogisticModel(
            scaler_mean=[float(value) for value in parameters["scaler_mean"]],
            scaler_scale=[float(value) for value in parameters["scaler_scale"]],
            coefficients=[
                float(value) for value in parameters["classifier_coefficients"]
            ],
            intercept=float(parameters["classifier_intercept"]),
        )
        return cls(model, model_version=artifact["model_version"])

    @classmethod
    def from_dataset(cls, data_path: str | Path = DEFAULT_DATA_PATH) -> "Predictor":
        path = Path(data_path)
        data = load_and_validate_dataset(path)
        splits = split_dataset(data)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                random_state=42,
                class_weight="balanced",
                max_iter=1000,
            ),
        )
        model.fit(splits.train[FEATURE_COLUMNS], splits.train["label"])
        dataset_digest = sha256(path.read_bytes()).hexdigest()[:12]
        return cls(model, model_version=f"logistic-regression-{dataset_digest}")

    def predict(self, *, feature_a: float, feature_b: float) -> Prediction:
        values = [float(feature_a), float(feature_b)]
        if not all(isfinite(value) for value in values):
            raise ValueError("Features must contain only finite values")

        features = pd.DataFrame([values], columns=FEATURE_COLUMNS)
        probabilities = self._model.predict_proba(features)[0]
        probability_by_label = {
            int(label): float(probability)
            for label, probability in zip(self._model.classes_, probabilities)
        }
        label = max(probability_by_label, key=probability_by_label.get)
        return Prediction(
            label=label,
            label_name=LABEL_NAMES[label],
            confidence=probability_by_label[label],
            malignant_probability=probability_by_label[0],
            benign_probability=probability_by_label[1],
            model_version=self.model_version,
        )
