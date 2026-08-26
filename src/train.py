"""Train and evaluate the Week 1 baseline classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = {"feature_a", "feature_b", "label"}


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
    if data["label"].nunique() < 2:
        raise ValueError("Dataset must contain at least two label classes")

    features = data[["feature_a", "feature_b"]]
    labels = data["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=42, stratify=labels
    )
    model = LogisticRegression(random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline classifier")
    parser.add_argument(
        "--data", default="data/sample.csv", help="Path to the CSV dataset"
    )
    args = parser.parse_args()

    try:
        metrics = train_and_evaluate(args.data)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1: {metrics['f1']:.3f}")


if __name__ == "__main__":
    main()
