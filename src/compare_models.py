"""Compare fixed baseline candidates without using the test split for selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.train import FEATURE_COLUMNS, load_and_validate_dataset, split_dataset


METRIC_NAMES = ("precision", "recall", "f1", "roc_auc")
MIN_MALIGNANT_PRECISION = 0.85
ModelFactory = Callable[[], Any]


def _default_model_factories() -> dict[str, ModelFactory]:
    return {
        "logistic_regression": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                random_state=42,
                class_weight="balanced",
                max_iter=1000,
            ),
        ),
        "decision_tree": lambda: DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
            max_depth=4,
        ),
        "random_forest": lambda: RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=3,
            n_jobs=1,
        ),
    }


MODEL_FACTORIES = _default_model_factories()


def _evaluate_model(model: Any, partition: pd.DataFrame) -> dict[str, Any]:
    features = partition[FEATURE_COLUMNS]
    labels = partition["label"]
    predictions = model.predict(features)
    malignant_class_index = list(model.classes_).index(0)
    malignant_scores = model.predict_proba(features)[:, malignant_class_index]
    malignant_labels = (labels == 0).astype(int)

    errors = []
    for index, actual, predicted in zip(partition.index, labels, predictions):
        if actual == predicted:
            continue
        subtype = (
            "malignant_false_negative" if actual == 0 else "benign_false_positive"
        )
        errors.append(
            {
                "category": "model",
                "subtype": subtype,
                "row_index": int(index),
                "actual_label": int(actual),
                "predicted_label": int(predicted),
            }
        )

    return {
        "metrics": {
            "precision": float(
                precision_score(labels, predictions, pos_label=0, zero_division=0)
            ),
            "recall": float(
                recall_score(labels, predictions, pos_label=0, zero_division=0)
            ),
            "f1": float(f1_score(labels, predictions, pos_label=0, zero_division=0)),
            "roc_auc": float(roc_auc_score(malignant_labels, malignant_scores)),
        },
        "errors": errors,
    }


def select_best_model(validation_results: Mapping[str, dict[str, Any]]) -> str:
    """Apply a precision floor, then rank by recall, F1, ROC-AUC, and name."""
    if not validation_results:
        raise ValueError("At least one validation result is required")

    eligible_models = [
        name
        for name, result in validation_results.items()
        if result["metrics"]["precision"] >= MIN_MALIGNANT_PRECISION
    ]
    if not eligible_models:
        raise ValueError(
            "No model meets the minimum malignant precision threshold of "
            f"{MIN_MALIGNANT_PRECISION:.2f}"
        )

    def ranking_key(name: str) -> tuple[float, float, float, float, str]:
        metrics = validation_results[name]["metrics"]
        return (
            -metrics["recall"],
            -metrics["f1"],
            -metrics["roc_auc"],
            -metrics["precision"],
            name,
        )

    # TODO: Replace this provisional precision floor with a stakeholder-approved
    # operating threshold and confidence-interval policy before deployment.
    return sorted(eligible_models, key=ranking_key)[0]


def compare_models(
    data_path: str | Path,
    *,
    model_factories: Mapping[str, ModelFactory] | None = None,
) -> dict[str, Any]:
    """Fit candidates on train, select on validation, then evaluate one test model."""
    factories = MODEL_FACTORIES if model_factories is None else model_factories
    if len(factories) < 3:
        raise ValueError("At least three model candidates are required")

    data = load_and_validate_dataset(data_path)
    splits = split_dataset(data)
    validation_results: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Any] = {}

    for name, factory in factories.items():
        model = factory()
        model.fit(splits.train[FEATURE_COLUMNS], splits.train["label"])
        fitted_models[name] = model
        validation_results[name] = _evaluate_model(model, splits.validation)

    selected_model = select_best_model(validation_results)
    test_result = _evaluate_model(fitted_models[selected_model], splits.test)

    return {
        "selection_policy": (
            f"validation precision >= {MIN_MALIGNANT_PRECISION:.2f}; then "
            "malignant recall, F1, ROC-AUC, precision"
        ),
        "split_rows": {
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
        },
        "validation": validation_results,
        "selected_model": selected_model,
        "test": test_result,
    }


def format_markdown_report(report: Mapping[str, Any]) -> str:
    """Render a compact, auditable comparison table."""
    lines = [
        "| Model | Precision | Recall | F1 | ROC-AUC | Model errors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in report["validation"].items():
        metrics = result["metrics"]
        lines.append(
            f"| {name} | {metrics['precision']:.3f} | {metrics['recall']:.3f} "
            f"| {metrics['f1']:.3f} | {metrics['roc_auc']:.3f} "
            f"| {len(result['errors'])} |"
        )

    test_metrics = report["test"]["metrics"]
    lines.extend(
        [
            "",
            (
                f"Selected model: `{report['selected_model']}` "
                f"({report['selection_policy']})."
            ),
            "",
            "| Test precision | Test recall | Test F1 | Test ROC-AUC | Model errors |",
            "|---:|---:|---:|---:|---:|",
            f"| {test_metrics['precision']:.3f} | {test_metrics['recall']:.3f} "
            f"| {test_metrics['f1']:.3f} | {test_metrics['roc_auc']:.3f} "
            f"| {len(report['test']['errors'])} |",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline model candidates")
    parser.add_argument(
        "--data",
        default="data/breast_cancer_wisconsin_diagnostic.csv",
        help="Path to the CSV dataset",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format",
    )
    args = parser.parse_args()

    try:
        report = compare_models(args.data)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(format_markdown_report(report))


if __name__ == "__main__":
    main()
