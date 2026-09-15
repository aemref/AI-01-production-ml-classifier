"""Verify the published inference artifact against its source dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.build_artifact import DEFAULT_DATA_PATH
from src.model_artifact import ArtifactValidationError, verify_artifact_dataset
from src.predictor import DEFAULT_ARTIFACT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify artifact integrity and source-dataset identity"
    )
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()

    try:
        artifact = verify_artifact_dataset(args.artifact, args.data)
    except (ArtifactValidationError, FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(f"Verified model: {artifact['model_version']}")
    print(f"Dataset SHA-256: {artifact['dataset']['sha256']}")


if __name__ == "__main__":
    main()
