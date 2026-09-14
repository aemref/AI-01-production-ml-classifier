from pathlib import Path

import pytest

from src.predictor import Predictor


REAL_DATA_PATH = (
    Path(__file__).parents[1] / "data" / "breast_cancer_wisconsin_diagnostic.csv"
)


@pytest.fixture(scope="module")
def predictor():
    return Predictor.from_dataset(REAL_DATA_PATH)


def test_predictor_returns_named_probabilities_and_version(predictor):
    prediction = predictor.predict(feature_a=17.99, feature_b=10.38)

    assert prediction.label in {0, 1}
    assert prediction.label_name in {"malignant", "benign"}
    assert prediction.confidence == pytest.approx(
        max(prediction.malignant_probability, prediction.benign_probability)
    )
    assert prediction.malignant_probability + prediction.benign_probability == (
        pytest.approx(1.0)
    )
    assert prediction.model_version.startswith("logistic-regression-")


def test_predictor_is_deterministic_for_the_same_input(predictor):
    first = predictor.predict(feature_a=13.54, feature_b=14.36)
    second = predictor.predict(feature_a=13.54, feature_b=14.36)

    assert first == second


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), -float("inf")])
def test_predictor_rejects_non_finite_features(predictor, invalid_value):
    with pytest.raises(ValueError, match="finite"):
        predictor.predict(feature_a=invalid_value, feature_b=10.0)
