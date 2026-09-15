import json
from pathlib import Path

import pytest

from src.model_artifact import (
    ArtifactValidationError,
    build_artifact,
    checksum_path_for,
    load_artifact,
    write_artifact,
)


REAL_DATA_PATH = (
    Path(__file__).parents[1] / "data" / "breast_cancer_wisconsin_diagnostic.csv"
)


@pytest.fixture(scope="module")
def artifact():
    return build_artifact(REAL_DATA_PATH)


def test_artifact_records_training_and_inference_contract(artifact):
    assert artifact["schema_version"] == 1
    assert artifact["model_version"].startswith("logistic-regression-")
    assert artifact["dataset"]["rows"] == 569
    assert artifact["training"]["train_rows"] == 398
    assert artifact["model"]["feature_columns"] == ["feature_a", "feature_b"]
    assert artifact["model"]["class_labels"] == [0, 1]


def test_written_artifact_round_trips_with_checksum(tmp_path, artifact):
    artifact_path = tmp_path / "model.json"

    digest = write_artifact(artifact, artifact_path)

    assert load_artifact(artifact_path) == artifact
    assert checksum_path_for(artifact_path).read_text().split() == [
        digest,
        "model.json",
    ]
    assert json.loads(artifact_path.read_text()) == artifact


def test_load_rejects_artifact_without_checksum(tmp_path, artifact):
    artifact_path = tmp_path / "model.json"
    artifact_path.write_text(json.dumps(artifact))

    with pytest.raises(ArtifactValidationError, match="checksum not found"):
        load_artifact(artifact_path)
