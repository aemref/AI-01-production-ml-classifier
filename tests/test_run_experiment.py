import json
from pathlib import Path

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
