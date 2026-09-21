from pathlib import Path
from types import SimpleNamespace

from src.mlflow_tracking import log_experiment_to_mlflow


class FakeRun:
    def __init__(self):
        self.info = SimpleNamespace(
            run_id="run-123",
            artifact_uri="file:///tmp/mlruns/run-123/artifacts",
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeMlflow:
    def __init__(self):
        self.calls = []

    def set_tracking_uri(self, value):
        self.calls.append(("set_tracking_uri", value))

    def set_experiment(self, value):
        self.calls.append(("set_experiment", value))

    def start_run(self, *, run_name):
        self.calls.append(("start_run", run_name))
        return FakeRun()

    def log_params(self, value):
        self.calls.append(("log_params", value))

    def log_metrics(self, value):
        self.calls.append(("log_metrics", value))

    def set_tags(self, value):
        self.calls.append(("set_tags", value))

    def log_artifact(self, path, *, artifact_path):
        self.calls.append(("log_artifact", path, artifact_path))


def _record():
    return {
        "schema_version": 1,
        "dataset": {"filename": "data.csv", "sha256": "abc123", "rows": 10},
        "experiment": {
            "selection_policy": "validation only",
            "split_rows": {"train": 6, "validation": 2, "test": 2},
            "validation": {
                "model_a": {
                    "metrics": {"precision": 0.9, "recall": 0.8},
                    "errors": [],
                }
            },
            "selected_model": "model_a",
            "test": {"metrics": {"precision": 0.7, "recall": 0.6}, "errors": []},
        },
    }


def test_logs_parameters_metrics_tags_and_report_artifact(tmp_path):
    report = tmp_path / "report.json"
    report.write_text("{}\n")
    fake_mlflow = FakeMlflow()

    result = log_experiment_to_mlflow(
        _record(),
        report,
        tracking_uri="file:./mlruns",
        experiment_name="ai01-local",
        run_name="baseline",
        mlflow_module=fake_mlflow,
    )

    assert result == {
        "artifact_uri": "file:///tmp/mlruns/run-123/artifacts",
        "experiment_name": "ai01-local",
        "run_id": "run-123",
        "tracking_uri": "file:./mlruns",
    }
    assert ("set_tracking_uri", "file:./mlruns") in fake_mlflow.calls
    assert ("set_experiment", "ai01-local") in fake_mlflow.calls
    assert ("start_run", "baseline") in fake_mlflow.calls
    assert (
        "log_params",
        {
            "dataset.filename": "data.csv",
            "dataset.sha256": "abc123",
            "dataset.rows": 10,
            "split.train_rows": 6,
            "split.validation_rows": 2,
            "split.test_rows": 2,
            "selection.policy": "validation only",
            "selection.model": "model_a",
        },
    ) in fake_mlflow.calls
    assert (
        "log_metrics",
        {
            "validation.model_a.precision": 0.9,
            "validation.model_a.recall": 0.8,
            "test.precision": 0.7,
            "test.recall": 0.6,
        },
    ) in fake_mlflow.calls
    assert ("log_artifact", str(report.resolve()), "reports") in fake_mlflow.calls
