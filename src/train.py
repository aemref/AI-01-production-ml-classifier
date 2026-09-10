"""Train and evaluate the baseline classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = {"feature_a", "feature_b", "label"}
FEATURE_COLUMNS = ["feature_a", "feature_b"]
EXPECTED_LABELS = {0, 1}
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


class DatasetSplits(NamedTuple):
    """Index-preserving train, validation, and test partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_dataset(
    data: pd.DataFrame,
    *,
    validation_size: float = VALIDATION_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = 42,
) -> DatasetSplits:
    """Create stratified, pairwise-disjoint train/validation/test partitions."""
    holdout_size = validation_size + test_size
    if validation_size <= 0 or test_size <= 0 or holdout_size >= 1:
        raise ValueError(
            "Validation and test sizes must be greater than 0 and sum to less than 1"
        )

    try:
        train, holdout = train_test_split(
            data,
            test_size=holdout_size,
            random_state=random_state,
            stratify=data["label"],
        )
        relative_test_size = test_size / holdout_size
        validation, test = train_test_split(
            holdout,
            test_size=relative_test_size,
            random_state=random_state,
            stratify=holdout["label"],
        )
    except ValueError as error:
        raise ValueError(
            "Dataset is too small for a stratified train/validation/test split; "
            "provide enough rows from both classes for all three partitions"
        ) from error

    return DatasetSplits(train=train, validation=validation, test=test)


def _evaluate_partition(
    model: LogisticRegression, partition: pd.DataFrame
) -> dict[str, float]:
    labels = partition["label"]
    predictions = model.predict(partition[FEATURE_COLUMNS])
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "malignant_recall": float(
            recall_score(labels, predictions, pos_label=0, zero_division=0)
        ),
    }


def train_and_evaluate(data_path: str | Path) -> dict[str, float]:
    """Fit on train only and return separate validation and test metrics."""
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if data.empty:
        raise ValueError("Dataset must contain at least one row")
    if data[list(REQUIRED_COLUMNS)].isnull().any().any():
        raise ValueError("Dataset must not contain empty values")

    features = data[FEATURE_COLUMNS]
    has_non_numeric_feature = any(
        not pd.api.types.is_numeric_dtype(features[column]) for column in features
    )
    if has_non_numeric_feature:
        raise ValueError("Feature columns must contain only numeric values")
    if features.isin([float("inf"), float("-inf")]).any().any():
        raise ValueError("Feature columns must contain only finite values")

    labels = set(data["label"].unique())
    if labels != EXPECTED_LABELS:
        raise ValueError("Label column must contain both binary classes 0 and 1")

    splits = split_dataset(data)
    model = LogisticRegression(random_state=42, class_weight="balanced")
    model.fit(splits.train[FEATURE_COLUMNS], splits.train["label"])

    # TODO: Use validation metrics for explicit candidate selection when Week 4
    # introduces more than one model. The test partition must remain untouched.
    validation_metrics = _evaluate_partition(model, splits.validation)
    test_metrics = _evaluate_partition(model, splits.test)

    return {
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_f1": validation_metrics["f1"],
        "validation_malignant_recall": validation_metrics["malignant_recall"],
        **test_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline classifier")
    parser.add_argument(
        "--data",
        default="data/breast_cancer_wisconsin_diagnostic.csv",
        help="Path to the CSV dataset",
    )
    args = parser.parse_args()

    try:
        metrics = train_and_evaluate(args.data)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(f"Validation accuracy: {metrics['validation_accuracy']:.3f}")
    print(f"Validation F1: {metrics['validation_f1']:.3f}")
    print(
        "Validation malignant recall: "
        f"{metrics['validation_malignant_recall']:.3f}"
    )
    print(f"Test accuracy: {metrics['accuracy']:.3f}")
    print(f"Test F1: {metrics['f1']:.3f}")
    print(f"Test malignant recall: {metrics['malignant_recall']:.3f}")


if __name__ == "__main__":
    main()
