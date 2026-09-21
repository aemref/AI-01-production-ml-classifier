"""Optional MLflow adapter for portable experiment records."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping


def _tracked_parameters(record: Mapping[str, Any]) -> dict[str, Any]:
    experiment = record["experiment"]
    return {
        "dataset.filename": record["dataset"]["filename"],
        "dataset.sha256": record["dataset"]["sha256"],
        "dataset.rows": record["dataset"]["rows"],
        "split.train_rows": experiment["split_rows"]["train"],
        "split.validation_rows": experiment["split_rows"]["validation"],
        "split.test_rows": experiment["split_rows"]["test"],
        "selection.policy": experiment["selection_policy"],
        "selection.model": experiment["selected_model"],
    }


def _tracked_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    experiment = record["experiment"]
    metrics = {
        f"validation.{model_name}.{metric_name}": float(value)
        for model_name, result in experiment["validation"].items()
        for metric_name, value in result["metrics"].items()
    }
    metrics.update(
        {
            f"test.{metric_name}": float(value)
            for metric_name, value in experiment["test"]["metrics"].items()
        }
    )
    return metrics


def log_experiment_to_mlflow(
    record: Mapping[str, Any],
    record_path: str | Path,
    *,
    tracking_uri: str,
    experiment_name: str,
    run_name: str | None = None,
    mlflow_module: Any | None = None,
) -> dict[str, str]:
    """Log one record to MLflow and return stable run identifiers."""
    mlflow = (
        importlib.import_module("mlflow") if mlflow_module is None else mlflow_module
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_params(_tracked_parameters(record))
        mlflow.log_metrics(_tracked_metrics(record))
        mlflow.set_tags(
            {
                "record.schema_version": str(record["schema_version"]),
                "run.source": "src.run_experiment",
            }
        )
        mlflow.log_artifact(str(Path(record_path).resolve()), artifact_path="reports")
        return {
            "artifact_uri": active_run.info.artifact_uri,
            "experiment_name": experiment_name,
            "run_id": active_run.info.run_id,
            "tracking_uri": tracking_uri,
        }
