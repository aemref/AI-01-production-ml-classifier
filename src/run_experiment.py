"""Run the fixed model comparison and emit a portable experiment record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.compare_models import compare_models
from src.experiment_record import build_experiment_record, write_experiment_record


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
    args = parser.parse_args()

    try:
        record, output_path = run_experiment(args.data, args.output)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "dataset_sha256": record["dataset"]["sha256"],
                "output": str(output_path),
                "selected_model": record["experiment"]["selected_model"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
