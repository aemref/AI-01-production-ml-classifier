from pathlib import Path

import pytest

from src.train import train_and_evaluate


DATA_PATH = Path(__file__).parents[1] / "data" / "sample.csv"
REAL_DATA_PATH = (
    Path(__file__).parents[1] / "data" / "breast_cancer_wisconsin_diagnostic.csv"
)


def test_baseline_returns_valid_metrics():
    metrics = train_and_evaluate(DATA_PATH)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert metrics["accuracy"] > 0.0


def test_real_dataset_supports_the_main_training_flow():
    metrics = train_and_evaluate(REAL_DATA_PATH)

    assert metrics["accuracy"] >= 0.85
    assert metrics["f1"] >= 0.85


def test_empty_dataset_has_clear_error(tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("feature_a,feature_b,label\n")

    with pytest.raises(ValueError, match="at least one row"):
        train_and_evaluate(empty_file)


def test_missing_columns_have_clear_error(tmp_path):
    invalid_file = tmp_path / "invalid.csv"
    invalid_file.write_text("feature_a,label\n1,0\n2,1\n")

    with pytest.raises(ValueError, match="required columns"):
        train_and_evaluate(invalid_file)


def test_non_numeric_feature_has_clear_error(tmp_path):
    invalid_file = tmp_path / "non_numeric.csv"
    invalid_file.write_text(
        "feature_a,feature_b,label\n"
        "bad,1,0\n10,2,0\n20,3,1\n21,4,1\n22,5,1\n"
    )

    with pytest.raises(ValueError, match="numeric"):
        train_and_evaluate(invalid_file)


def test_non_finite_feature_has_clear_error(tmp_path):
    invalid_file = tmp_path / "non_finite.csv"
    invalid_file.write_text(
        "feature_a,feature_b,label\n"
        "1,inf,0\n10,2,0\n20,3,1\n21,4,1\n22,5,1\n"
    )

    with pytest.raises(ValueError, match="finite"):
        train_and_evaluate(invalid_file)


def test_too_few_rows_for_stratified_split_has_clear_error(tmp_path):
    boundary_file = tmp_path / "too_small.csv"
    boundary_file.write_text(
        "feature_a,feature_b,label\n1,1,0\n2,2,0\n3,3,1\n4,4,1\n"
    )

    with pytest.raises(ValueError, match="too small"):
        train_and_evaluate(boundary_file)
