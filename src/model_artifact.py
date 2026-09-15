"""Build and validate a portable artifact for the selected baseline model."""

from __future__ import annotations

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.train import FEATURE_COLUMNS, load_and_validate_dataset, split_dataset


ARTIFACT_SCHEMA_VERSION = 1
MODEL_TYPE = "standard-scaler-logistic-regression"
DEFAULT_RANDOM_STATE = 42
EXPECTED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "model_version",
    "dataset",
    "training",
    "model",
}


class ArtifactValidationError(ValueError):
    """Raised when an artifact cannot be trusted by the inference runtime."""


def checksum_path_for(artifact_path: str | Path) -> Path:
    path = Path(artifact_path)
    return path.with_suffix(f"{path.suffix}.sha256")


def build_artifact(data_path: str | Path) -> dict[str, Any]:
    """Fit the selected model on train only and return its portable parameters."""
    path = Path(data_path)
    data = load_and_validate_dataset(path)
    splits = split_dataset(data, random_state=DEFAULT_RANDOM_STATE)
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=DEFAULT_RANDOM_STATE,
            class_weight="balanced",
            max_iter=1000,
        ),
    )
    pipeline.fit(splits.train[FEATURE_COLUMNS], splits.train["label"])
    scaler: StandardScaler = pipeline.named_steps["standardscaler"]
    classifier: LogisticRegression = pipeline.named_steps["logisticregression"]
    dataset_checksum = sha256(path.read_bytes()).hexdigest()

    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_version": f"logistic-regression-{dataset_checksum[:12]}",
        "dataset": {
            "sha256": dataset_checksum,
            "rows": len(data),
        },
        "training": {
            "random_state": DEFAULT_RANDOM_STATE,
            "validation_size": 0.15,
            "test_size": 0.15,
            "train_rows": len(splits.train),
        },
        "model": {
            "type": MODEL_TYPE,
            "feature_columns": FEATURE_COLUMNS,
            "class_labels": [int(value) for value in classifier.classes_],
            "scaler_mean": [float(value) for value in scaler.mean_],
            "scaler_scale": [float(value) for value in scaler.scale_],
            "classifier_coefficients": [
                float(value) for value in classifier.coef_[0]
            ],
            "classifier_intercept": float(classifier.intercept_[0]),
        },
    }


def serialize_artifact(artifact: Mapping[str, Any]) -> bytes:
    """Return a stable, human-reviewable representation of an artifact."""
    validate_artifact(artifact)
    return (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_artifact(artifact: Mapping[str, Any], artifact_path: str | Path) -> str:
    """Write an artifact and its SHA-256 sidecar, returning the checksum."""
    path = Path(artifact_path)
    contents = serialize_artifact(artifact)
    digest = sha256(contents).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    checksum_path_for(path).write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def load_artifact(artifact_path: str | Path) -> dict[str, Any]:
    """Verify the checksum, parse JSON, and enforce the runtime contract."""
    path = Path(artifact_path)
    try:
        contents = path.read_bytes()
    except FileNotFoundError as error:
        raise ArtifactValidationError(f"Model artifact not found: {path}") from error

    checksum_path = checksum_path_for(path)
    try:
        checksum_parts = checksum_path.read_text(encoding="utf-8").split()
    except FileNotFoundError as error:
        raise ArtifactValidationError(
            f"Model artifact checksum not found: {checksum_path}"
        ) from error
    if len(checksum_parts) != 2 or checksum_parts[1] != path.name:
        raise ArtifactValidationError("Model artifact checksum file is malformed")

    actual_checksum = sha256(contents).hexdigest()
    if checksum_parts[0] != actual_checksum:
        raise ArtifactValidationError("Model artifact checksum mismatch")

    try:
        artifact = json.loads(contents)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError("Model artifact is not valid UTF-8 JSON") from error
    validate_artifact(artifact)
    return artifact


def verify_artifact_dataset(
    artifact_path: str | Path, data_path: str | Path
) -> dict[str, Any]:
    """Verify artifact integrity and bind it to the current validated dataset."""
    artifact = load_artifact(artifact_path)
    dataset_path = Path(data_path)
    data = load_and_validate_dataset(dataset_path)
    actual_checksum = sha256(dataset_path.read_bytes()).hexdigest()
    if artifact["dataset"]["sha256"] != actual_checksum:
        raise ArtifactValidationError(
            "Model artifact dataset checksum does not match the current dataset"
        )
    if artifact["dataset"]["rows"] != len(data):
        raise ArtifactValidationError(
            "Model artifact row count does not match the current dataset"
        )
    return artifact


def _finite_number_list(value: Any, *, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != len(FEATURE_COLUMNS):
        raise ArtifactValidationError(
            f"Model artifact {field} must contain {len(FEATURE_COLUMNS)} values"
        )
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ArtifactValidationError(f"Model artifact {field} must be numeric")
    numbers = [float(item) for item in value]
    if not all(isfinite(item) for item in numbers):
        raise ArtifactValidationError(f"Model artifact {field} must be finite")
    return numbers


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject incompatible or incomplete artifact metadata and parameters."""
    if not isinstance(artifact, Mapping):
        raise ArtifactValidationError("Model artifact root must be an object")
    if set(artifact) != EXPECTED_TOP_LEVEL_FIELDS:
        raise ArtifactValidationError("Model artifact fields are incompatible")
    if artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactValidationError("Unsupported model artifact schema version")
    version = artifact.get("model_version")
    if not isinstance(version, str) or not version.startswith("logistic-regression-"):
        raise ArtifactValidationError("Model artifact version is invalid")

    dataset = artifact.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ArtifactValidationError("Model artifact dataset metadata is missing")
    dataset_checksum = dataset.get("sha256")
    if (
        not isinstance(dataset_checksum, str)
        or len(dataset_checksum) != 64
        or any(character not in "0123456789abcdef" for character in dataset_checksum)
    ):
        raise ArtifactValidationError("Model artifact dataset checksum is invalid")
    if version != f"logistic-regression-{dataset_checksum[:12]}":
        raise ArtifactValidationError("Model artifact version does not match its dataset")
    dataset_rows = dataset.get("rows")
    if isinstance(dataset_rows, bool) or not isinstance(dataset_rows, int):
        raise ArtifactValidationError("Model artifact dataset row count is invalid")
    if dataset_rows <= 0:
        raise ArtifactValidationError("Model artifact dataset row count is invalid")

    training = artifact.get("training")
    if not isinstance(training, Mapping):
        raise ArtifactValidationError("Model artifact training metadata is missing")
    expected_training = {
        "random_state": DEFAULT_RANDOM_STATE,
        "validation_size": 0.15,
        "test_size": 0.15,
    }
    for field, expected in expected_training.items():
        if training.get(field) != expected:
            raise ArtifactValidationError(
                f"Model artifact training field {field} is incompatible"
            )
    train_rows = training.get("train_rows")
    if (
        isinstance(train_rows, bool)
        or not isinstance(train_rows, int)
        or train_rows <= 0
        or train_rows >= dataset_rows
    ):
        raise ArtifactValidationError("Model artifact training row count is invalid")

    model = artifact.get("model")
    if not isinstance(model, Mapping):
        raise ArtifactValidationError("Model artifact parameters are missing")
    if model.get("type") != MODEL_TYPE:
        raise ArtifactValidationError("Unsupported model artifact type")
    if model.get("feature_columns") != FEATURE_COLUMNS:
        raise ArtifactValidationError("Model artifact feature contract is incompatible")
    if model.get("class_labels") != [0, 1]:
        raise ArtifactValidationError("Model artifact class contract is incompatible")

    _finite_number_list(model.get("scaler_mean"), field="scaler mean")
    scale = _finite_number_list(model.get("scaler_scale"), field="scaler scale")
    if any(value <= 0 for value in scale):
        raise ArtifactValidationError("Model artifact scaler values must be positive")
    _finite_number_list(
        model.get("classifier_coefficients"), field="classifier coefficients"
    )
    intercept = model.get("classifier_intercept")
    if (
        isinstance(intercept, bool)
        or not isinstance(intercept, (int, float))
        or not isfinite(float(intercept))
    ):
        raise ArtifactValidationError("Model artifact classifier intercept is invalid")
