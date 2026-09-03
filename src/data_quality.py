"""Reproducible missingness, imbalance, and leakage-risk checks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {"feature_a", "feature_b", "label"}
FEATURE_COLUMNS = ["feature_a", "feature_b"]
# TODO: Calibrate these heuristic thresholds with domain owners before production use.
HIGH_CORRELATION_THRESHOLD = 0.95
MODERATE_IMBALANCE_RATIO = 1.5
SEVERE_IMBALANCE_RATIO = 4.0


def _imbalance_level(ratio: float | None) -> str:
    if ratio is None:
        return "undefined"
    if ratio >= SEVERE_IMBALANCE_RATIO:
        return "severe"
    if ratio >= MODERATE_IMBALANCE_RATIO:
        return "moderate"
    return "low"


def analyze_data_quality(data_path: str | Path) -> dict[str, Any]:
    """Return deterministic data-quality and leakage-risk indicators for a CSV."""
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {names}")
    if data.empty:
        raise ValueError("Dataset must contain at least one row")

    selected = data[[*FEATURE_COLUMNS, "label"]]
    missing_by_column = {
        column: int(count) for column, count in selected.isna().sum().items()
    }
    observed_labels = set(data["label"].dropna().unique())
    if not observed_labels.issubset({0, 1}):
        raise ValueError("Label column may contain only binary classes 0 and 1")
    class_counts = {
        int(label): int(count)
        for label, count in data["label"].dropna().value_counts().sort_index().items()
    }
    if len(class_counts) >= 2:
        majority_to_minority_ratio = max(class_counts.values()) / min(
            class_counts.values()
        )
    else:
        majority_to_minority_ratio = None

    numeric_features = [
        column
        for column in FEATURE_COLUMNS
        if pd.api.types.is_numeric_dtype(data[column])
    ]
    target_copy_features: list[str] = []
    if not data["label"].isna().any():
        for column in numeric_features:
            feature = data[column]
            if feature.isna().any():
                continue
            is_direct_copy = feature.eq(data["label"]).all()
            is_inverse_copy = feature.eq(1 - data["label"]).all()
            if is_direct_copy or is_inverse_copy:
                target_copy_features.append(column)

    target_correlations: dict[str, float] = {}
    high_target_correlation_features: list[str] = []
    if pd.api.types.is_numeric_dtype(data["label"]):
        for column in numeric_features:
            correlation = data[[column, "label"]].corr().iloc[0, 1]
            if pd.isna(correlation):
                continue
            target_correlations[column] = float(correlation)
            if (
                column not in target_copy_features
                and abs(float(correlation)) >= HIGH_CORRELATION_THRESHOLD
            ):
                high_target_correlation_features.append(column)

    duplicate_mask = data.duplicated(subset=FEATURE_COLUMNS, keep=False)
    duplicate_feature_rows = int(duplicate_mask.sum())
    conflicting_duplicate_groups = 0
    if duplicate_feature_rows:
        duplicate_groups = data.loc[duplicate_mask].groupby(
            FEATURE_COLUMNS, dropna=False
        )["label"]
        conflicting_duplicate_groups = int((duplicate_groups.nunique() > 1).sum())

    return {
        "row_count": int(len(data)),
        "missing_by_column": missing_by_column,
        "missing_total": sum(missing_by_column.values()),
        "class_counts": class_counts,
        "majority_to_minority_ratio": majority_to_minority_ratio,
        "imbalance_level": (
            "single-class"
            if len(class_counts) < 2
            else _imbalance_level(majority_to_minority_ratio)
        ),
        "non_numeric_features": sorted(set(FEATURE_COLUMNS) - set(numeric_features)),
        "target_copy_features": target_copy_features,
        "target_correlations": target_correlations,
        "high_target_correlation_features": high_target_correlation_features,
        "duplicate_feature_rows": duplicate_feature_rows,
        "conflicting_duplicate_groups": conflicting_duplicate_groups,
    }


def format_report(report: dict[str, Any]) -> str:
    """Format the quality indicators for a compact experiment log."""
    counts = ", ".join(
        f"{label}={count}" for label, count in report["class_counts"].items()
    )
    ratio = report["majority_to_minority_ratio"]
    ratio_text = "undefined" if ratio is None else f"{ratio:.3f}"
    correlations = ", ".join(
        f"{feature}={value:.3f}"
        for feature, value in report["target_correlations"].items()
    )
    return "\n".join(
        [
            f"Rows: {report['row_count']}",
            f"Missing values: {report['missing_total']}",
            f"Class counts: {counts or 'none'}",
            f"Majority/minority ratio: {ratio_text} ({report['imbalance_level']})",
            f"Non-numeric features: {report['non_numeric_features'] or 'none'}",
            f"Target-copy features: {report['target_copy_features'] or 'none'}",
            f"Feature/target correlations: {correlations or 'none'}",
            "High target-correlation features: "
            f"{report['high_target_correlation_features'] or 'none'}",
            f"Duplicate feature rows: {report['duplicate_feature_rows']}",
            "Conflicting duplicate groups: "
            f"{report['conflicting_duplicate_groups']}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze dataset quality risks")
    parser.add_argument(
        "--data",
        default="data/breast_cancer_wisconsin_diagnostic.csv",
        help="Path to the CSV dataset",
    )
    args = parser.parse_args()

    try:
        report = analyze_data_quality(args.data)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(format_report(report))


if __name__ == "__main__":
    main()
