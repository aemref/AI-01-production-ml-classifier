"""Reusable prediction service for the selected baseline classifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.train import FEATURE_COLUMNS, load_and_validate_dataset, split_dataset


DEFAULT_DATA_PATH = Path("data/breast_cancer_wisconsin_diagnostic.csv")
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


class Predictor:
    """Fit and serve the validation-selected logistic regression baseline."""

    def __init__(self, model: Any, *, model_version: str) -> None:
        self._model = model
        self.model_version = model_version

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
