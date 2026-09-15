import json
from hashlib import sha256
from pathlib import Path

import pytest

from src.model_artifact import (
    ArtifactValidationError,
    build_artifact,
    checksum_path_for,
    load_artifact,
    verify_artifact_dataset,
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


def test_load_rejects_artifact_tampering(tmp_path, artifact):
    artifact_path = tmp_path / "model.json"
    write_artifact(artifact, artifact_path)
    artifact_path.write_text(
        artifact_path.read_text().replace("-3.100854428737666", "3.100854428737666")
    )

    with pytest.raises(ArtifactValidationError, match="checksum mismatch"):
        load_artifact(artifact_path)


def test_load_rejects_rechecksummed_incompatible_contract(tmp_path, artifact):
    artifact_path = tmp_path / "model.json"
    incompatible = json.loads(json.dumps(artifact))
    incompatible["model"]["feature_columns"] = ["feature_b", "feature_a"]
    contents = (json.dumps(incompatible, indent=2, sort_keys=True) + "\n").encode()
    artifact_path.write_bytes(contents)
    checksum_path_for(artifact_path).write_text(
        f"{sha256(contents).hexdigest()}  {artifact_path.name}\n"
    )

    with pytest.raises(ArtifactValidationError, match="feature contract"):
        load_artifact(artifact_path)


def test_artifact_verification_binds_the_model_to_its_dataset(tmp_path, artifact):
    artifact_path = tmp_path / "model.json"
    write_artifact(artifact, artifact_path)

    verified = verify_artifact_dataset(artifact_path, REAL_DATA_PATH)

    assert verified["model_version"] == artifact["model_version"]


def test_artifact_verification_rejects_a_different_dataset(tmp_path, artifact):
    artifact_path = tmp_path / "model.json"
    changed_data_path = tmp_path / "changed.csv"
    write_artifact(artifact, artifact_path)
    changed_data_path.write_bytes(
        REAL_DATA_PATH.read_bytes().replace(b"17.99", b"17.98", 1)
    )

    with pytest.raises(ArtifactValidationError, match="dataset checksum"):
        verify_artifact_dataset(artifact_path, changed_data_path)
