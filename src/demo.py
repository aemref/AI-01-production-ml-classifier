"""Run a deterministic, local prediction demo from the published artifact."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from src import __version__
from src.predictor import Predictor


DEMO_CASES = (
    {"name": "high-radius example", "feature_a": 17.99, "feature_b": 10.38},
    {"name": "low-radius example", "feature_a": 9.504, "feature_b": 12.44},
)


def build_demo_report(predictor: Predictor) -> dict[str, object]:
    """Return the versioned results for fixed, dataset-derived feature examples."""

    predictions = []
    for case in DEMO_CASES:
        result = predictor.predict(
            feature_a=float(case["feature_a"]),
            feature_b=float(case["feature_b"]),
        )
        predictions.append({**case, "prediction": result.to_dict()})
    return {
        "application_version": __version__,
        "model_version": predictor.model_version,
        "disclaimer": "Educational example only; not for diagnosis or treatment.",
        "predictions": predictions,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed examples through the checksummed model artifact."
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit one-line JSON instead of indented JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_demo_report(Predictor.from_artifact())
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
