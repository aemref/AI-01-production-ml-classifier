"""Command-line builder for the versioned inference artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.model_artifact import build_artifact, write_artifact


DEFAULT_DATA_PATH = Path("data/breast_cancer_wisconsin_diagnostic.csv")
DEFAULT_ARTIFACT_DIRECTORY = Path("artifacts")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the selected classifier's portable model artifact"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the validated training CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Artifact JSON path; defaults to artifacts/<model-version>.json",
    )
    args = parser.parse_args()

    try:
        artifact = build_artifact(args.data)
        output_path = args.output or (
            DEFAULT_ARTIFACT_DIRECTORY / f"{artifact['model_version']}.json"
        )
        checksum = write_artifact(artifact, output_path)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(f"Model version: {artifact['model_version']}")
    print(f"Artifact: {output_path}")
    print(f"SHA-256: {checksum}")


if __name__ == "__main__":
    main()
