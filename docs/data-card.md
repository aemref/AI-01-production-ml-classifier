# Data Card: Breast Cancer Wisconsin (Diagnostic) Subset

## Summary

This repository includes a 569-row, two-feature derivative of the Breast Cancer
Wisconsin (Diagnostic) dataset for a binary classification learning exercise.
It contains 212 malignant and 357 benign observations and no missing values.

| Repository column | Source field | Type | Meaning |
| --- | --- | --- | --- |
| `feature_a` | `mean radius` | float | Mean distance from center to perimeter points |
| `feature_b` | `mean texture` | float | Standard deviation of gray-scale values |
| `label` | `target` / diagnosis | integer | `0`: malignant; `1`: benign |

## Motivation and intended use

The subset replaces a synthetic fixture in the repository's main path while
keeping the Week 2 integration intentionally small. It is suitable for learning,
pipeline tests, input-validation exercises, and reproducible baseline
experiments.

It must not be used for medical diagnosis, treatment decisions, patient
screening, or claims of clinical effectiveness. The repository's model and
evaluation have not been clinically validated.

## Source, attribution, and license

- Creators: William Wolberg, Olvi Mangasarian, Nick Street, and W. Street.
- Source: UCI Machine Learning Repository.
- Citation: Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993),
  *Breast Cancer Wisconsin (Diagnostic)*, DOI
  [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B).
- Canonical record:
  https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- License: CC BY 4.0. See [`data/LICENSE.md`](../data/LICENSE.md).

The repository copy was exported from scikit-learn's bundled
`load_breast_cancer` representation. Only `mean radius`, `mean texture`, and
`target` were retained, then renamed to the repository's existing CSV contract.
No rows were sampled, synthesized, or relabeled.

## Composition and quality checks

- Rows: 569, with one observation per digitized fine-needle aspirate image.
- Features used: 2 numeric columns from the 30 source features.
- Class balance: 212 malignant (`0`), 357 benign (`1`).
- Missing values in the included columns: 0.
- Duplicate and outlier analysis: not yet performed; tracked as future EDA.
- File SHA-256:
  `98b12889accbae788456d9442fa0753d2d15a361241b543f9fe21795176d1303`.

The training path rejects missing columns, empty data, missing or non-finite
feature values, non-numeric features, labels other than the complete `{0, 1}`
set, and inputs too small for the stratified split.

## Evaluation and limitations

The baseline uses a fixed 75/25 stratified split and Logistic Regression. With
random seed 42, the included subset produces Accuracy `0.888` and benign-class
F1 `0.912`. These results are regression signals, not estimates of clinical
utility.

Important limitations include the historical collection context, unknown
fitness for current or broader populations, reduced feature set, class
imbalance, a single split, lack of subgroup metadata, and no external
validation. Label-`1` F1 emphasizes benign cases and can obscure malignant-case
errors. Future work should add confusion matrices, per-class metrics,
cross-validation, duplicate/outlier analysis, and bias assessment.

## Maintenance

The dataset is checked into the repository for offline reproducibility. Any
replacement must update this card, attribution/change notice, row and class
counts, checksum, regression expectations, and license review.
