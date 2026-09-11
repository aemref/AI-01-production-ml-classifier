# Week 3: Baseline Model Comparison

Tracking issue: [#4 — Week 3: Baseline model and measurement](https://github.com/aemref/AI-01-production-ml-classifier/issues/4)

## Goal and vertical slice

Build on the leakage-safe train/validation/test pipeline by comparing three
fixed baseline candidates with malignant-class precision, recall, F1, and
ROC-AUC. Fit on train only, select on validation only, and evaluate only the
selected candidate on the untouched test partition.

Acceptance criteria: three named candidates report all four metrics; candidate
fits contain only train indices; normal, minimum-boundary, and failure fixtures
are covered; errors are categorized; commands and results are reproducible; a
model-card draft records intended use and limitations.

## Contract

- **Input:** licensed two-feature Breast Cancer Wisconsin subset with numeric
  `feature_a`, `feature_b`, and both binary labels; versioned normal, boundary,
  and failure fixtures.
- **Expected output:** validation comparison table, validation-only selection,
  one selected-model test result, and row-indexed model-error records.
- **Metrics:** precision, recall, F1, and ROC-AUC with malignant `label=0` treated
  as the positive class. The provisional selection gate requires validation
  precision of at least `0.85`, then ranks recall, F1, ROC-AUC, and precision.
- **Dependencies:** Python 3.11, pandas, scikit-learn, and pytest. Runtime needs
  no network, API key, or external download.
- **Risks:** two features cannot represent full diagnostic context; the dataset
  is small and imbalanced; a single split has high variance; fixed hyperparameters
  may overfit; test-guided iteration would leak; outputs could be medically
  misused. This is a learning baseline, not a diagnostic system.

## Test-first evidence

The comparison tests and three fixtures were written before the implementation.
The first focused run failed during collection as expected:

```text
ModuleNotFoundError: No module named 'src.compare_models'
1 error
```

Normal input requires three complete metric sets and categorized model errors.
The 12-row boundary fixture verifies the smallest supported stratified split.
The single-class failure fixture must produce the existing clear binary-label
error. Additional regression tests prove every candidate fits train indices
only and that unsafe selection fails clearly.

## Starting measurement and critical fix

The initial recall-first policy selected Decision Tree because validation recall
was `1.000`, despite validation precision of `0.821`. Its untouched-test
precision was `0.763` and ROC-AUC was `0.894`. Using the test result to choose a
different model would itself leak, so the correction is validation-only: reject
candidates below the provisional `0.85` malignant-precision floor, then rank
eligible models by recall, F1, ROC-AUC, and precision.

The regression test first failed with `high_recall` selected instead of the
precision-safe candidate. After the correction, Logistic Regression is selected
without consulting test labels. The threshold remains a visible `TODO` pending
stakeholder approval and confidence-interval analysis.

## Result table

Command: `python -m src.compare_models`

| Validation model | Precision | Recall | F1 | ROC-AUC | Model errors |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.879 | 0.906 | 0.892 | 0.977 | 7 |
| Decision Tree | 0.821 | 1.000 | 0.901 | 0.967 | 7 |
| Random Forest | 0.833 | 0.938 | 0.882 | 0.976 | 8 |

Selected model: Logistic Regression.

| Test precision | Test recall | Test F1 | Test ROC-AUC | Model errors |
|---:|---:|---:|---:|---:|
| 0.784 | 0.906 | 0.841 | 0.948 | 11 |

Validation error breakdown: Logistic Regression has 3 malignant false negatives
and 4 benign false positives; Decision Tree has 0 and 7; Random Forest has 2 and
6. The selected model's test errors contain 3 malignant false negatives and 8
benign false positives. Machine-readable output is available with `--format json`.

## Failure classification

| Category | Observed example | Handling |
|---|---|---|
| Data | Single-class, target-copy, duplicate-feature, and missing-value fixtures | Rejected before split with a clear validation error. |
| Code | Missing comparison module in the intentional test-first run | Implemented and covered by regression tests. |
| Model | Malignant false negatives and benign false positives | Row index, actual label, predicted label, and subtype are emitted as `model` errors. |
| Prompt | Not applicable; no prompt or generative model exists in this pipeline | Recorded as out of scope rather than silently conflated with model error. |
| Infrastructure | No failure in the local or clean-environment verification | CI repeats tests and all three executable checks. |
| Usage | Missing dataset path | Raises `FileNotFoundError` with the requested path. |

## Verification commands

```bash
python -m pytest -q
python src/train.py
python -m src.compare_models
python src/data_quality.py
python -m compileall -q src tests
```

The comparison uses fixed random seed `42` and reports split sizes `398/85/86`.
CI runs the same model-comparison command on every push and pull request.

Clean-environment verification used a new temporary Python 3.14 virtual
environment because Python 3.11 is not installed locally. Installing only
`requirements.txt` produced `32 passed`; training and data-quality commands
exited `0`. A separate, uncontended comparison benchmark completed in `0.91s`
wall time (`0.82s` user, `0.08s` system). GitHub Actions provides the required
independent Python 3.11 compatibility check.
