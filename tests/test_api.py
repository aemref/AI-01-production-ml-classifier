import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.model_artifact import ArtifactValidationError
from src.predictor import Predictor


REAL_DATA_PATH = (
    Path(__file__).parents[1] / "data" / "breast_cancer_wisconsin_diagnostic.csv"
)


@pytest.fixture(scope="module")
def client():
    predictor = Predictor.from_dataset(REAL_DATA_PATH)
    with TestClient(create_app(predictor)) as test_client:
        yield test_client


def test_default_app_loads_the_persisted_artifact():
    with TestClient(create_app()) as artifact_client:
        health = artifact_client.get("/health")

    assert health.status_code == 200
    assert health.json()["model_version"] == "logistic-regression-98b12889accb"


def test_app_fails_closed_when_artifact_integrity_is_missing(tmp_path):
    artifact_path = tmp_path / "untrusted.json"
    artifact_path.write_text("{}")

    with pytest.raises(ArtifactValidationError, match="checksum not found"):
        with TestClient(create_app(artifact_path=artifact_path)):
            pass


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
    assert response.json()["error"]["code"] == "invalid_request"
    assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]
    assert response.json()["error"]["details"][0]["location"] == [
        "body",
        "feature_b",
    ]


def test_predict_rejects_unknown_fields(client):
    response = client.post(
        "/predict",
        json={"feature_a": 17.99, "feature_b": 10.38, "diagnosis": "unknown"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_failure_log_is_structured_and_excludes_payload(caplog, client):
    caplog.set_level("WARNING", logger="classifier.api")

    response = client.post(
        "/predict",
        json={"feature_a": "sensitive-invalid-value", "feature_b": 10.38},
    )

    record = next(
        record for record in caplog.records if "request_rejected" in record.message
    )
    logged = json.loads(record.message)
    assert logged == {
        "event": "request_rejected",
        "request_id": response.headers["X-Request-ID"],
        "path": "/predict",
        "status_code": 422,
        "error_code": "invalid_request",
    }
    assert "sensitive-invalid-value" not in record.message


def test_each_response_has_a_unique_request_id(client):
    first = client.get("/health")
    second = client.get("/health")

    assert len(first.headers["X-Request-ID"]) == 32
    assert first.headers["X-Request-ID"] != second.headers["X-Request-ID"]
