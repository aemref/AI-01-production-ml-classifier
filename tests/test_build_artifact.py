import subprocess
import sys
from pathlib import Path

from src.model_artifact import load_artifact


ROOT = Path(__file__).parents[1]
REAL_DATA_PATH = ROOT / "data" / "breast_cancer_wisconsin_diagnostic.csv"


def test_cli_builds_a_loadable_versioned_artifact(tmp_path):
    output_path = tmp_path / "baseline.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.build_artifact",
            "--data",
            str(REAL_DATA_PATH),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    artifact = load_artifact(output_path)
    assert artifact["model_version"] in result.stdout
    assert "SHA-256:" in result.stdout


def test_cli_reports_a_missing_dataset_without_a_traceback(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.build_artifact",
            "--data",
            str(tmp_path / "missing.csv"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Dataset not found" in result.stderr
    assert "Traceback" not in result.stderr
