import json

from src import __version__
from src.demo import DEMO_CASES, build_demo_report, main
from src.predictor import Predictor


def test_demo_uses_fixed_examples_and_published_artifact():
    report = build_demo_report(Predictor.from_artifact())

    assert report["application_version"] == __version__
    assert report["model_version"] == "logistic-regression-98b12889accb"
    assert len(report["predictions"]) == len(DEMO_CASES) == 2
    assert {
        item["prediction"]["label_name"] for item in report["predictions"]
    } == {"malignant", "benign"}
    for item in report["predictions"]:
        prediction = item["prediction"]
        assert prediction["model_version"] == report["model_version"]
        assert 0.0 <= prediction["confidence"] <= 1.0


def test_demo_cli_emits_machine_readable_json(capsys):
    assert main(["--compact"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["model_version"] == "logistic-regression-98b12889accb"
    assert report["disclaimer"].startswith("Educational")
