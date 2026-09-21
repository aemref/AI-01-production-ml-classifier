import hashlib
from pathlib import Path

import json

from src.experiment_record import (
    build_experiment_record,
    sha256_file,
    write_experiment_record,
)


def _comparison_report():
    return {
        "selection_policy": "validation only",
        "split_rows": {"train": 6, "validation": 2, "test": 2},
        "validation": {},
        "selected_model": "example",
        "test": {"metrics": {}},
    }


def test_sha256_file_reads_the_file_bytes(tmp_path):
    dataset = tmp_path / "dataset.csv"
    content = b"feature,label\n1,0\n"
    dataset.write_bytes(content)

    assert sha256_file(dataset) == hashlib.sha256(content).hexdigest()


def test_experiment_record_identifies_dataset_without_exposing_local_path(tmp_path):
    dataset = tmp_path / "private" / "dataset.csv"
    dataset.parent.mkdir()
    dataset.write_text("feature,label\n1,0\n")

    record = build_experiment_record(dataset, _comparison_report())

    assert record["schema_version"] == 1
    assert record["dataset"] == {
        "filename": "dataset.csv",
        "sha256": sha256_file(dataset),
        "rows": 10,
    }
    assert str(dataset.parent) not in repr(record)
    assert record["experiment"]["selected_model"] == "example"


def test_write_experiment_record_creates_stable_json(tmp_path):
    output = tmp_path / "nested" / "record.json"
    record = build_experiment_record(__file__, _comparison_report())

    written_path = write_experiment_record(record, output)

    assert written_path == output.resolve()
    assert json.loads(output.read_text()) == record
    assert output.read_text().endswith("\n")
