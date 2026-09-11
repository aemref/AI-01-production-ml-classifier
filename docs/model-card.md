# Model Card Draft: Two-Feature Breast Cancer Baseline

Status: draft, educational baseline, not approved for clinical or production use.

## Model details

The selected candidate is a scikit-learn pipeline containing `StandardScaler`
and class-balanced Logistic Regression (`random_state=42`, `max_iter=1000`). It
was selected from Logistic Regression, a depth-4 Decision Tree, and a constrained
Random Forest using the fixed validation partition only.

## Intended use

This model demonstrates reproducible data validation, leakage-safe splitting,
baseline comparison, and metric reporting. It may support learning and software
regression tests. It must not diagnose, triage, recommend treatment, or substitute
for clinicians and validated medical devices.

## Data

The repository contains a licensed, transformed subset of the Breast Cancer
Wisconsin (Diagnostic) dataset: 569 rows, mean radius and mean texture features,
212 malignant labels (`0`), and 357 benign labels (`1`). Provenance, license,
transformations, and quality findings are documented in `docs/data-card.md`.

## Evaluation

The deterministic stratified split contains 398 train, 85 validation, and 86
test rows. Models fit train only. Candidate selection requires provisional
malignant precision `>=0.85` on validation and then ranks malignant recall, F1,
ROC-AUC, and precision. Only the selected Logistic Regression is evaluated on
test.

| Partition | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Validation | 0.879 | 0.906 | 0.892 | 0.977 |
| Test | 0.784 | 0.906 | 0.841 | 0.948 |

All metrics treat malignant `label=0` as positive. Test errors comprise 3
malignant false negatives and 8 benign false positives.

## Limitations and ethical considerations

- The two-feature representation omits most original diagnostic measurements.
- One historical dataset and one fixed split cannot establish generalization,
  calibration, fairness, robustness, or present-day clinical validity.
- No subgroup attributes are available for bias analysis.
- The provisional precision floor is not a clinical operating threshold.
- False negatives may miss malignant cases; false positives may cause needless
  alarm. Neither risk is acceptable without expert requirements and validation.
- Exact target copies and duplicate feature rows are blocked, but structural
  checks cannot prove the absence of causal or collection-time leakage.

## Reproducibility and monitoring

Run `python -m src.compare_models` from a fresh environment installed with
`requirements.txt`. CI executes the full test suite, training command, comparison,
and data-quality audit. Future work should add repeated cross-validation,
confidence intervals, calibration, subgroup evaluation, versioned model artifacts,
drift monitoring, and stakeholder-approved thresholds before any deployment.
