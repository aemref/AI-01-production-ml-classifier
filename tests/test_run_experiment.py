import json
from pathlib import Path

import src.run_experiment as run_experiment_module
from src.run_experiment import run_experiment


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "comparison_normal.csv"


def test_run_experiment_writes_a_reproducible_comparison_record(tmp_path):
    output = tmp_path / "record.json"

    record, written_path = run_experiment(FIXTURE_PATH, output)

    assert written_path == output.resolve()
    assert json.loads(output.read_text()) == record
    assert record["dataset"]["filename"] == FIXTURE_PATH.name
    assert record["dataset"]["rows"] == 24
    assert record["experiment"]["selected_model"] in {
        "logistic_regression",
        "decision_tree",
        "random_forest",
    }


def test_cli_tracks_only_when_explicitly_requested(monkeypatch, tmp_path, capsys):
    output = tmp_path / "record.json"
    calls = []

    def fake_log(record, record_path, **settings):
        calls.append((record, record_path, settings))
        return {
            "artifact_uri": "file:///tmp/artifacts",
            "experiment_name": settings["experiment_name"],
            "run_id": "run-123",
            "tracking_uri": settings["tracking_uri"],
        }

    monkeypatch.setattr(run_experiment_module, "log_experiment_to_mlflow", fake_log)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_experiment",
            "--data",
            str(FIXTURE_PATH),
            "--output",
            str(output),
            "--track-mlflow",
            "--run-name",
            "test-run",
        ],
    )

    run_experiment_module.main()

    response = json.loads(capsys.readouterr().out)
    assert response["mlflow"]["run_id"] == "run-123"
    assert calls[0][1] == output.resolve()
    assert calls[0][2] == {
        "tracking_uri": "sqlite:///mlruns.db",
        "experiment_name": "ai01-production-ml-classifier",
        "run_name": "test-run",
    }
