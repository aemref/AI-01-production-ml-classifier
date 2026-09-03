from pathlib import Path

import pytest

from src.data_quality import analyze_data_quality


REAL_DATA_PATH = (
    Path(__file__).parents[1] / "data" / "breast_cancer_wisconsin_diagnostic.csv"
)


def test_real_dataset_quality_report_is_reproducible():
    report = analyze_data_quality(REAL_DATA_PATH)

    assert report["row_count"] == 569
    assert report["missing_total"] == 0
    assert report["class_counts"] == {0: 212, 1: 357}
    assert report["majority_to_minority_ratio"] == pytest.approx(357 / 212)
    assert report["target_copy_features"] == []
    assert report["target_correlations"] == pytest.approx(
        {"feature_a": -0.730029, "feature_b": -0.415185}, abs=1e-6
    )
    assert report["high_target_correlation_features"] == []


def test_report_detects_missing_values_imbalance_and_target_copy(tmp_path):
    risky_file = tmp_path / "risky.csv"
    risky_file.write_text(
        "feature_a,feature_b,label\n"
        "0,10,0\n0,,0\n1,20,1\n1,21,1\n1,22,1\n1,23,1\n"
    )

    report = analyze_data_quality(risky_file)

    assert report["missing_by_column"] == {
        "feature_a": 0,
        "feature_b": 1,
        "label": 0,
    }
    assert report["majority_to_minority_ratio"] == 2.0
    assert report["target_copy_features"] == ["feature_a"]


def test_empty_dataset_has_clear_analysis_error(tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("feature_a,feature_b,label\n")

    with pytest.raises(ValueError, match="at least one row"):
        analyze_data_quality(empty_file)


def test_missing_columns_have_clear_analysis_error(tmp_path):
    invalid_file = tmp_path / "invalid.csv"
    invalid_file.write_text("feature_a,label\n1,0\n2,1\n")

    with pytest.raises(ValueError, match="required columns"):
        analyze_data_quality(invalid_file)


def test_invalid_label_has_clear_analysis_error(tmp_path):
    invalid_file = tmp_path / "invalid_label.csv"
    invalid_file.write_text(
        "feature_a,feature_b,label\n1,1,0\n2,2,1\n3,3,unknown\n"
    )

    with pytest.raises(ValueError, match="binary classes 0 and 1"):
        analyze_data_quality(invalid_file)
