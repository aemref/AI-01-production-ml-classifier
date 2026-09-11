from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.compare_models import (
    METRIC_NAMES,
    compare_models,
    format_markdown_report,
    select_best_model,
)
from src.train import split_dataset


FIXTURE_PATH = Path(__file__).parent / "fixtures"
REAL_DATA_PATH = (
    Path(__file__).parents[1] / "data" / "breast_cancer_wisconsin_diagnostic.csv"
)
EXPECTED_MODELS = {"logistic_regression", "decision_tree", "random_forest"}


class SpyClassifier:
    def __init__(self, fitted_indices, expected_labels):
        self.fitted_indices = fitted_indices
        self.expected_labels = expected_labels
        self.classes_ = np.array([0, 1])

    def fit(self, features, labels):
        self.fitted_indices.append(set(features.index))
        return self

    def predict(self, features):
        return self.expected_labels.loc[features.index].to_numpy()

    def predict_proba(self, features):
        malignant = np.where(self.predict(features) == 0, 0.75, 0.25)
        return np.column_stack([malignant, 1 - malignant])


def test_normal_fixture_compares_three_models_and_reports_metrics():
    report = compare_models(FIXTURE_PATH / "comparison_normal.csv")

    assert set(report["validation"]) == EXPECTED_MODELS
    assert report["selected_model"] in EXPECTED_MODELS
    for result in report["validation"].values():
        assert set(result["metrics"]) == set(METRIC_NAMES)
        assert all(0.0 <= value <= 1.0 for value in result["metrics"].values())
        assert all(error["category"] == "model" for error in result["errors"])
    assert set(report["test"]["metrics"]) == set(METRIC_NAMES)


def test_minimum_split_boundary_fixture_supports_comparison():
    report = compare_models(FIXTURE_PATH / "comparison_boundary.csv")

    assert len(report["validation"]) == 3
    assert report["test"]["metrics"]["roc_auc"] >= 0.0


def test_single_class_failure_fixture_has_clear_error():
    with pytest.raises(ValueError, match="both binary classes"):
        compare_models(FIXTURE_PATH / "comparison_failure_single_class.csv")


def test_missing_input_is_classified_as_a_clear_usage_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        compare_models(tmp_path / "missing.csv")


def test_all_candidates_fit_only_the_train_partition():
    data = pd.read_csv(REAL_DATA_PATH)
    expected_train_indices = set(split_dataset(data).train.index)
    fitted_indices = []
    factories = {
        name: (
            lambda indices=fitted_indices, labels=data["label"]: SpyClassifier(
                indices, labels
            )
        )
        for name in EXPECTED_MODELS
    }

    compare_models(REAL_DATA_PATH, model_factories=factories)

    assert fitted_indices == [expected_train_indices] * 3


def test_selection_rejects_recall_winner_below_precision_floor():
    validation_results = {
        "high_f1": {
            "metrics": {
                "precision": 1.0,
                "recall": 0.8,
                "f1": 0.9,
                "roc_auc": 1.0,
            }
        },
        "high_recall": {
            "metrics": {
                "precision": 0.7,
                "recall": 1.0,
                "f1": 0.8,
                "roc_auc": 0.9,
            }
        },
    }

    assert select_best_model(validation_results) == "high_f1"


def test_selection_has_clear_error_when_no_model_meets_precision_floor():
    validation_results = {
        "unsafe": {
            "metrics": {
                "precision": 0.4,
                "recall": 1.0,
                "f1": 0.6,
                "roc_auc": 0.9,
            }
        }
    }

    with pytest.raises(ValueError, match="minimum malignant precision"):
        select_best_model(validation_results)


def test_markdown_report_contains_auditable_table_and_error_counts():
    report = compare_models(FIXTURE_PATH / "comparison_normal.csv")

    rendered = format_markdown_report(report)

    assert "| Model | Precision | Recall | F1 | ROC-AUC |" in rendered
    assert "Selected model:" in rendered
    assert "Model errors" in rendered
