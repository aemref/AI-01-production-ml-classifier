"""Verify deterministic repository artifacts required for the v1 release."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

from src import __version__
from src.model_artifact import ArtifactValidationError, verify_artifact_dataset


ROOT = Path(__file__).parents[1]
EXPECTED_VERSION = "1.0.0"
MODEL_ARTIFACT = Path("artifacts/logistic-regression-98b12889accb.json")
DATASET = Path("data/breast_cancer_wisconsin_diagnostic.csv")
REQUIRED_DOCUMENTS = {
    Path("README.md"): ("Reproducible Demo", "not a medical"),
    Path("CHANGELOG.md"): ("1.0.0", "not clinically validated"),
    Path("docs/architecture.md"): ("Trust and failure boundaries",),
    Path("docs/data-card.md"): ("License",),
    Path("docs/model-card.md"): ("not approved for clinical",),
    Path("docs/release-checklist.md"): ("Final verification commands",),
}


@dataclass(frozen=True)
class ReleaseCheck:
    name: str
    passed: bool
    detail: str


def _check_demo_gif(root: Path) -> ReleaseCheck:
    asset = root / "docs" / "assets" / "demo.gif"
    try:
        content = asset.read_bytes()
    except FileNotFoundError:
        return ReleaseCheck("demo_gif", False, "docs/assets/demo.gif is missing")
    passed = content.startswith((b"GIF87a", b"GIF89a")) and content.count(
        b"\x21\xf9\x04"
    ) >= 3
    return ReleaseCheck(
        "demo_gif",
        passed,
        "animated GIF has at least three frames" if passed else "invalid demo GIF",
    )


def _check_documents(root: Path) -> ReleaseCheck:
    problems = []
    for relative_path, markers in REQUIRED_DOCUMENTS.items():
        document = root / relative_path
        if not document.is_file():
            problems.append(f"missing {relative_path}")
            continue
        text = document.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                problems.append(f"{relative_path} lacks {marker!r}")
    return ReleaseCheck(
        "release_documents",
        not problems,
        "; ".join(problems) if problems else "required release evidence is present",
    )


def run_release_checks(
    *, root: Path = ROOT, expected_version: str = EXPECTED_VERSION
) -> list[ReleaseCheck]:
    """Run repository-state checks that complement tests and container checks."""

    checks = [
        ReleaseCheck(
            "application_version",
            __version__ == expected_version,
            f"expected {expected_version}; found {__version__}",
        )
    ]
    try:
        artifact = verify_artifact_dataset(root / MODEL_ARTIFACT, root / DATASET)
    except (ArtifactValidationError, FileNotFoundError, ValueError) as error:
        checks.append(ReleaseCheck("model_artifact", False, str(error)))
    else:
        checks.append(
            ReleaseCheck(
                "model_artifact",
                True,
                f"verified {artifact['model_version']}",
            )
        )
    checks.extend((_check_demo_gif(root), _check_documents(root)))
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise ValueError("release readiness does not accept positional arguments")
    checks = run_release_checks()
    ready = all(check.passed for check in checks)
    print(
        json.dumps(
            {
                "release": f"v{EXPECTED_VERSION}",
                "ready": ready,
                "checks": [asdict(check) for check in checks],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
