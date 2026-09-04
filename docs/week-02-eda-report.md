# Week 2 EDA and Evaluation Report

## Goal

Publish the Week 2 EDA findings in English and fix the most consequential
evaluation weakness with a regression test.

## Test scenarios

| Scenario | Fixture | Expected behavior |
| --- | --- | --- |
| Normal | `tests/fixtures/normal.csv` | Training completes and all metrics are bounded in `[0, 1]`. |
| Boundary | `tests/fixtures/boundary_minimum_split.csv` | The smallest supported stratified split completes. |
| Failure | `tests/fixtures/failure_missing_value.csv` | Training rejects the missing feature with a clear error. |

The real-data regression scenario also requires Accuracy `>= 0.85`, benign F1
`>= 0.85`, and malignant recall `>= 0.90`.

## Success signal and baseline

Malignant recall is the primary signal because a malignant sample predicted as
benign is the most consequential model error in this educational setting.

| Signal | Original baseline | Target | Class-balanced result |
| --- | ---: | ---: | ---: |
| Accuracy | 0.888 | Preserve `>= 0.85` | 0.888 |
| Benign F1 | 0.912 | Preserve `>= 0.85` | 0.908 |
| Malignant recall | 0.830 | `>= 0.90` | 0.906 |
| Malignant predicted benign | 9 | Reduce | 5 |
| Benign predicted malignant | 7 | Observe trade-off | 11 |

## EDA findings

- The checked subset has 569 rows, two numeric features, and no missing values.
- Labels contain 212 malignant and 357 benign observations. The 1.684
  majority/minority ratio is moderate under the project's temporary heuristic.
- There are no duplicate feature vectors, conflicting duplicates, exact target
  copies, or features above the temporary absolute-correlation warning of 0.95.
- Feature/target correlations are -0.730 for mean radius and -0.415 for mean
  texture. Association is expected here and does not by itself establish
  leakage.
- Tukey's 1.5xIQR rule flags 14 mean-radius and 7 mean-texture rows. They are
  retained because range checks show plausible values and outliers may contain
  useful class signal.

The executable analysis and saved outputs are in
`notebooks/week-02-eda.ipynb`. Dataset provenance, license, and permitted use
remain documented in `docs/data-card.md` and `data/LICENSE.md`.

## Failure classification

| Category | Observation | Disposition |
| --- | --- | --- |
| Data | No missing, invalid, duplicate, or obvious leakage rows in the repository dataset. | No row removal or imputation. |
| Code | The public evaluation result exposed Accuracy and benign F1 but not malignant recall. | Add malignant recall to the API, CLI, and tests. |
| Model | The original model predicted 9 malignant test rows as benign. | Use balanced class weights; reduced to 5. |
| Prompt | No prompt or generative model participates in this pipeline. | Not applicable. |
| Infrastructure | Local commands completed without infrastructure failure. | CI repeats tests, training, and audit. |
| Usage | Medical use would be unsafe and outside the data card's intended use. | Keep the clinical-use warning prominent. |

## Critical fix and regression protection

The model now uses `class_weight="balanced"`. This is the smallest change that
addresses the observed minority-class weakness: malignant recall improves from
0.830 to 0.906 without reducing Accuracy. The existing benign F1 moves from
0.912 to 0.908, and false positives increase, so the trade-off is stated rather
than hidden.

`test_real_dataset_supports_the_main_training_flow` now fails if malignant
recall drops below 0.90. It was written first and initially failed with
`KeyError: 'malignant_recall'`, demonstrating that the old evaluation contract
did not expose the required signal.

## Verification commands

```bash
python -m pytest -q
python src/train.py
python src/data_quality.py
python -m compileall -q src tests
```

## Clean-environment results

The commands ran with Python 3.14.7 in a fresh temporary virtual environment
created only from `requirements.txt`.

| Command | Result |
| --- | --- |
| `python -m pytest -q` | `15 passed in 0.84s` |
| `python src/train.py` | Accuracy `0.888`; benign F1 `0.908`; malignant recall `0.906` |
| `python src/data_quality.py` | 569 rows; 0 missing; ratio `1.684`; no obvious leakage flag |
| `python -m compileall -q src tests` | Passed with no output |
