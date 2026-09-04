# Week 2: Missingness, Imbalance, and Leakage Risks

## Goal

Analyze missing-data, class-imbalance, and data-leakage risks.

## Scope

In scope: deterministic checks for missing values, class counts and imbalance,
direct target copies, unusually high feature/target correlation, and duplicate
feature vectors; unit tests, CI execution, and a recorded interpretation.

Out of scope: automatic imputation/resampling, model tuning, full 30-feature EDA,
causal leakage analysis, clinical validation, and production threshold policy.

## Test-first evidence

`tests/test_data_quality.py` was added before the implementation. Its first run
failed during collection as expected:

```text
ModuleNotFoundError: No module named 'src.data_quality'
1 error
```

The fixture defines the intended behavior: report real-data statistics, detect
a missing value and a feature that copies the target, and return clear errors
for empty or schema-invalid inputs.

## Reproduction

```bash
.venv/bin/python -m pytest -q
.venv/bin/python src/data_quality.py
```

Observed analysis:

```text
Rows: 569
Missing values: 0
Class counts: 0=212, 1=357
Majority/minority ratio: 1.684 (moderate)
Non-numeric features: none
Target-copy features: none
Feature/target correlations: feature_a=-0.730, feature_b=-0.415
High target-correlation features: none
Duplicate feature rows: 0
Conflicting duplicate groups: 0
```

## Findings and decisions

### Missing data

No value is missing in the checked repository subset, so no imputation is
introduced. The training path continues to reject missing values rather than
silently changing them. This is appropriate for the current checked-in fixture;
future external data needs a train-only imputation policy.

### Class imbalance

There are 357 benign and 212 malignant rows: 62.7% versus 37.3%, a
majority/minority ratio of `1.684`. This is meaningful but not extreme. The
existing stratified split is retained, and no resampling or class weighting is
added without comparative evidence. Accuracy and benign-class F1 can hide
malignant errors, so per-class recall and a confusion matrix remain required
follow-up work.

### Leakage

Neither feature is an exact/inverse target copy; neither exceeds the temporary
absolute correlation warning threshold of `0.95`; and there are no duplicate
feature vectors that could cross the random split. Code inspection also confirms
that model fitting occurs after the split and there is currently no learned
preprocessing fitted on all rows.

These checks cover obvious structural leakage only. They do not establish that
the historical collection process, feature definitions, or future joins are
leak-free. The hard-coded correlation and imbalance thresholds are explicitly
marked `TODO` in `src/data_quality.py` and require domain review before any
production use.

## Validation result

The original full-suite run completed with `12 passed`; the audit output matched
the record above, and the then-current baseline remained Accuracy `0.888` / F1
`0.912`. A later Week 2 evaluation pass introduced class balancing and
malignant-recall reporting; see `docs/week-02-eda-report.md` for that explicitly
versioned result. CI runs both the training baseline and this audit on every push
and pull request.
