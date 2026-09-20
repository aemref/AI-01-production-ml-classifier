from math import isclose

import pytest

from src.drift import feature_shift


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
