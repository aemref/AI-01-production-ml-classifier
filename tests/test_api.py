from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.predictor import Predictor


REAL_DATA_PATH = (
    Path(__file__).parents[1] / "data" / "breast_cancer_wisconsin_diagnostic.csv"
)


@pytest.fixture(scope="module")
def client():
    predictor = Predictor.from_dataset(REAL_DATA_PATH)
    with TestClient(create_app(predictor)) as test_client:
        yield test_client


def test_health_reports_loaded_model_version(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_version"].startswith("logistic-regression-")


def test_predict_returns_validated_probabilities(client):
    response = client.post(
        "/predict",
        json={"feature_a": 17.99, "feature_b": 10.38},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {0, 1}
    assert body["label_name"] in {"malignant", "benign"}
    assert body["confidence"] == pytest.approx(
        max(body["malignant_probability"], body["benign_probability"])
    )
    assert body["malignant_probability"] + body["benign_probability"] == (
        pytest.approx(1.0)
    )


def test_predict_rejects_missing_feature(client):
    response = client.post("/predict", json={"feature_a": 17.99})

    assert response.status_code == 422


def test_predict_rejects_unknown_fields(client):
    response = client.post(
        "/predict",
        json={"feature_a": 17.99, "feature_b": 10.38, "diagnosis": "unknown"},
    )

    assert response.status_code == 422
