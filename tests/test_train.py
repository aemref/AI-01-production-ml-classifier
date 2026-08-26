from pathlib import Path

import pytest

from src.train import train_and_evaluate


DATA_PATH = Path(__file__).parents[1] / "data" / "sample.csv"


def test_baseline_returns_valid_metrics():
    metrics = train_and_evaluate(DATA_PATH)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert metrics["accuracy"] > 0.0


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
