import json

from src.release_readiness import main, run_release_checks


def test_repository_release_evidence_is_complete():
    checks = run_release_checks()

    assert {check.name for check in checks} == {
        "application_version",
        "demo_gif",
        "model_artifact",
        "release_documents",
    }
    assert all(check.passed for check in checks), checks


def test_version_mismatch_blocks_release():
    checks = run_release_checks(expected_version="9.9.9")

    version = next(check for check in checks if check.name == "application_version")
    assert not version.passed
    assert "expected 9.9.9" in version.detail


def test_cli_emits_machine_readable_gate_results(capsys):
    assert main() == 0

    report = json.loads(capsys.readouterr().out)
    assert report["release"] == "v1.0.0"
    assert report["ready"] is True
    assert all(check["passed"] for check in report["checks"])
