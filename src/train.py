"""Train and evaluate the baseline classifier."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = {"feature_a", "feature_b", "label"}
FEATURE_COLUMNS = ["feature_a", "feature_b"]
EXPECTED_LABELS = {0, 1}
TEST_SIZE = 0.25


def train_and_evaluate(data_path: str | Path) -> dict[str, float]:
    """Train the baseline model and return accuracy and F1 metrics."""
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

    class_counts = data["label"].value_counts()
    test_rows = math.ceil(len(data) * TEST_SIZE)
    train_rows = len(data) - test_rows
    if class_counts.min() < 2 or min(test_rows, train_rows) < len(EXPECTED_LABELS):
        raise ValueError(
            "Dataset is too small for a stratified split; provide at least two "
            "rows per class and enough rows for both split partitions"
        )

    labels = data["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE, random_state=42, stratify=labels
    )
    model = LogisticRegression(random_state=42, class_weight="balanced")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "malignant_recall": float(
            recall_score(y_test, predictions, pos_label=0, zero_division=0)
        ),
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

    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1: {metrics['f1']:.3f}")
    print(f"Malignant recall: {metrics['malignant_recall']:.3f}")


if __name__ == "__main__":
    main()
