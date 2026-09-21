"""Run the fixed model comparison and emit a portable experiment record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.compare_models import compare_models
from src.experiment_record import build_experiment_record, write_experiment_record
from src.mlflow_tracking import log_experiment_to_mlflow


DEFAULT_DATA_PATH = Path("data/breast_cancer_wisconsin_diagnostic.csv")
DEFAULT_OUTPUT_PATH = Path("reports/model-comparison.json")


def run_experiment(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> tuple[dict[str, Any], Path]:
    """Compare models and persist the result with its dataset identity."""
    report = compare_models(data_path)
    record = build_experiment_record(data_path, report)
    written_path = write_experiment_record(record, output_path)
    return record, written_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and record the deterministic model comparison"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--track-mlflow",
        action="store_true",
        help="Log the completed record to the configured MLflow backend",
    )
    parser.add_argument(
        "--tracking-uri",
        default="file:./mlruns",
        help="MLflow tracking URI used only with --track-mlflow",
    )
    parser.add_argument(
        "--experiment-name",
        default="ai01-production-ml-classifier",
        help="MLflow experiment used only with --track-mlflow",
    )
    parser.add_argument(
        "--run-name",
        help="Optional MLflow run name used only with --track-mlflow",
    )
    args = parser.parse_args()

    try:
        record, output_path = run_experiment(args.data, args.output)
        mlflow_result = None
        if args.track_mlflow:
            mlflow_result = log_experiment_to_mlflow(
                record,
                output_path,
                tracking_uri=args.tracking_uri,
                experiment_name=args.experiment_name,
                run_name=args.run_name,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    result = {
        "dataset_sha256": record["dataset"]["sha256"],
        "output": str(output_path),
        "selected_model": record["experiment"]["selected_model"],
    }
    if mlflow_result is not None:
        result["mlflow"] = mlflow_result
    print(
        json.dumps(result, sort_keys=True)
    )


if __name__ == "__main__":
    main()
