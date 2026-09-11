from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.train import split_dataset, train_and_evaluate


DATA_PATH = Path(__file__).parents[1] / "data" / "sample.csv"
FIXTURE_PATH = Path(__file__).parent / "fixtures"
REAL_DATA_PATH = (
    Path(__file__).parents[1] / "data" / "breast_cancer_wisconsin_diagnostic.csv"
)


def test_train_validation_test_splits_are_disjoint_and_complete():
    data = pd.read_csv(REAL_DATA_PATH)

    splits = split_dataset(data)

    train_ids = set(splits.train.index)
    validation_ids = set(splits.validation.index)
    test_ids = set(splits.test.index)
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert train_ids | validation_ids | test_ids == set(data.index)
    split_lengths = (len(splits.train), len(splits.validation), len(splits.test))
    assert split_lengths == (398, 85, 86)
    assert all(set(partition["label"]) == {0, 1} for partition in splits)


def test_model_fit_receives_train_partition_only(monkeypatch):
    data = pd.read_csv(REAL_DATA_PATH)
    expected_train_ids = set(split_dataset(data).train.index)
    fitted_ids = set()
    original_fit = LogisticRegression.fit

    def record_fit(model, features, labels, *args, **kwargs):
        fitted_ids.update(features.index)
        return original_fit(model, features, labels, *args, **kwargs)

    monkeypatch.setattr(LogisticRegression, "fit", record_fit)

    train_and_evaluate(REAL_DATA_PATH)

    assert fitted_ids == expected_train_ids


def test_baseline_returns_valid_metrics():
    metrics = train_and_evaluate(DATA_PATH)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert metrics["accuracy"] > 0.0


def test_real_dataset_supports_the_main_training_flow():
    metrics = train_and_evaluate(REAL_DATA_PATH)

    assert metrics["validation_accuracy"] >= 0.85
    assert metrics["validation_f1"] >= 0.85
    assert metrics["validation_malignant_recall"] >= 0.90
    assert metrics["accuracy"] >= 0.85
    assert metrics["f1"] >= 0.85
    assert metrics["malignant_recall"] >= 0.90


def test_normal_scenario_fixture_completes_training():
    metrics = train_and_evaluate(FIXTURE_PATH / "normal.csv")

    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_minimum_viable_split_boundary_fixture_completes_training():
    metrics = train_and_evaluate(FIXTURE_PATH / "boundary_minimum_split.csv")

    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_missing_value_failure_fixture_has_clear_error():
    with pytest.raises(ValueError, match="empty values"):
        train_and_evaluate(FIXTURE_PATH / "failure_missing_value.csv")


def test_target_copy_failure_fixture_has_clear_leakage_error():
    with pytest.raises(ValueError, match="target leakage.*feature_a"):
        train_and_evaluate(FIXTURE_PATH / "failure_target_copy.csv")


def test_inverted_target_copy_has_clear_leakage_error(tmp_path):
    data = pd.read_csv(FIXTURE_PATH / "failure_target_copy.csv")
    data["feature_a"] = 1 - data["label"]
    inverted_copy_file = tmp_path / "inverted_target_copy.csv"
    data.to_csv(inverted_copy_file, index=False)

    with pytest.raises(ValueError, match="target leakage.*feature_a"):
        train_and_evaluate(inverted_copy_file)


def test_duplicate_feature_failure_fixture_has_clear_leakage_error():
    with pytest.raises(ValueError, match="split leakage.*duplicate feature rows"):
        train_and_evaluate(FIXTURE_PATH / "failure_duplicate_features.csv")


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


@pytest.mark.parametrize(
    ("validation_size", "test_size"),
    [(0.0, 0.15), (0.15, 0.0), (0.50, 0.50), (-0.1, 0.15)],
)
def test_invalid_split_fractions_have_clear_error(validation_size, test_size):
    data = pd.read_csv(REAL_DATA_PATH)

    with pytest.raises(ValueError, match="greater than 0 and sum to less than 1"):
        split_dataset(data, validation_size=validation_size, test_size=test_size)
