"""Deterministic feature-shift diagnostics for the educational classifier."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Sequence

from src.model_artifact import verify_artifact_dataset
from src.predictor import DEFAULT_ARTIFACT_PATH, Predictor
from src.train import FEATURE_COLUMNS, load_and_validate_dataset, split_dataset


def feature_shift(reference: Sequence[float], candidate: Sequence[float]) -> dict:
    """Measure signed mean change in units of the reference population SD.

    A constant reference cannot define a standardized change. Return ``None``
    in that case instead of emitting infinity into JSON reports.
    """
    if not reference or not candidate:
        raise ValueError("Feature samples must be nonempty")
    values = [*reference, *candidate]
    if not all(isfinite(float(value)) for value in values):
        raise ValueError("Feature samples must contain only finite values")

    reference_mean = sum(reference) / len(reference)
    candidate_mean = sum(candidate) / len(candidate)
    reference_sd = (
        sum((value - reference_mean) ** 2 for value in reference) / len(reference)
    ) ** 0.5
    difference = candidate_mean - reference_mean
    standardized = difference / reference_sd if reference_sd else None
    return {
        "reference_mean": reference_mean,
        "candidate_mean": candidate_mean,
        "reference_sd": reference_sd,
        "mean_change": difference,
        "standardized_mean_change": standardized,
    }


def simulate_drift(
    data_path: str | Path,
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    shift_sd: float = 1.0,
) -> dict:
    """Shift test-set radius by reference SD and compare paired predictions.

    The original test labels are intentionally not scored on shifted rows: this
    synthetic perturbation does not provide ground-truth outcomes.
    """
    if not isfinite(shift_sd) or shift_sd < 0:
        raise ValueError("shift_sd must be a finite nonnegative number")
    artifact = verify_artifact_dataset(artifact_path, data_path)
    data = load_and_validate_dataset(data_path)
    test = split_dataset(data).test
    baseline = test[FEATURE_COLUMNS].copy()
    reference_radius = baseline["feature_a"].tolist()
    radius_sd = feature_shift(reference_radius, reference_radius)["reference_sd"]
    shifted = baseline.copy()
    shifted["feature_a"] += shift_sd * radius_sd

    predictor = Predictor.from_artifact(artifact_path)
    before = [
        predictor.predict(feature_a=row.feature_a, feature_b=row.feature_b).label
        for row in baseline.itertuples(index=False)
    ]
    after = [
        predictor.predict(feature_a=row.feature_a, feature_b=row.feature_b).label
        for row in shifted.itertuples(index=False)
    ]
    return {
        "dataset_sha256": artifact["dataset"]["sha256"],
        "model_version": predictor.model_version,
        "test_rows": len(test),
        "simulation": {
            "feature": "feature_a",
            "shift_sd": shift_sd,
            "added_radius_units": shift_sd * radius_sd,
        },
        "features": {
            column: feature_shift(baseline[column].tolist(), shifted[column].tolist())
            for column in FEATURE_COLUMNS
        },
        "predictions": {
            "baseline_malignant_count": before.count(0),
            "shifted_malignant_count": after.count(0),
            "changed_count": sum(old != new for old, new in zip(before, after)),
        },
    }
