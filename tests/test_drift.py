from math import isclose
from pathlib import Path

import pytest

from src.drift import feature_shift, simulate_drift


REAL_DATA_PATH = (
    Path(__file__).parents[1] / "data" / "breast_cancer_wisconsin_diagnostic.csv"
)


def test_feature_shift_uses_reference_scale_and_preserves_direction():
    result = feature_shift([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])

    assert result["reference_mean"] == 2.0
    assert result["candidate_mean"] == 3.0
    assert result["mean_change"] == 1.0
    assert isclose(result["standardized_mean_change"], 1.224744871, rel_tol=1e-8)


def test_constant_reference_avoids_nonfinite_json_metrics():
    result = feature_shift([2.0, 2.0], [3.0, 3.0])

    assert result["reference_sd"] == 0.0
    assert result["standardized_mean_change"] is None


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [([], [1.0]), ([1.0], []), ([float("inf")], [1.0])],
)
def test_invalid_samples_are_rejected(reference, candidate):
    with pytest.raises(ValueError, match="nonempty|finite"):
        feature_shift(reference, candidate)


def test_simulation_is_paired_reproducible_and_changes_only_radius():
    first = simulate_drift(REAL_DATA_PATH, shift_sd=1.0)
    second = simulate_drift(REAL_DATA_PATH, shift_sd=1.0)

    assert first == second
    assert first["test_rows"] == 86
    assert isclose(first["features"]["feature_a"]["standardized_mean_change"], 1.0)
    assert first["features"]["feature_b"]["mean_change"] == 0.0
    assert first["predictions"]["changed_count"] > 0


def test_zero_shift_keeps_all_paired_predictions_unchanged():
    report = simulate_drift(REAL_DATA_PATH, shift_sd=0.0)

    assert report["predictions"]["changed_count"] == 0
    assert report["predictions"]["baseline_malignant_count"] == report[
        "predictions"
    ]["shifted_malignant_count"]


@pytest.mark.parametrize("shift", [-1.0, float("inf"), float("nan")])
def test_simulation_rejects_invalid_shift(shift):
    with pytest.raises(ValueError, match="finite nonnegative"):
        simulate_drift(REAL_DATA_PATH, shift_sd=shift)
