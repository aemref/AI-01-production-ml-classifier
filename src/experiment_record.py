"""Build deterministic, portable records for model-comparison experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


EXPERIMENT_RECORD_SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest for a local file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_experiment_record(
    data_path: str | Path,
    comparison_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach immutable dataset identity to a model-comparison report."""
    path = Path(data_path)
    return {
        "schema_version": EXPERIMENT_RECORD_SCHEMA_VERSION,
        "dataset": {
            "filename": path.name,
            "sha256": sha256_file(path),
            "rows": sum(comparison_report["split_rows"].values()),
        },
        "experiment": dict(comparison_report),
    }
