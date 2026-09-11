# AI-01 Production ML Classifier

An **AI/ML + production engineering** learning project. It contains a small,
testable classification pipeline and the repository practices needed to evolve
it safely.

## Problem

The project classifies breast mass measurements as malignant or benign using a
small, reproducible subset of the real Breast Cancer Wisconsin (Diagnostic)
dataset. It exists to establish a baseline for data validation, model training,
testing, and CI; it is a learning project, not a medical diagnostic tool.

## Approach

The baseline uses Logistic Regression from scikit-learn. The pipeline reads a
CSV file, validates its required columns and values, creates stratified
train/validation/test partitions with a fixed random seed, fits a class-balanced
model on train only, and reports separate validation and test metrics.

The default dataset contains the mean radius and mean texture features. Its
provenance, transformations, appropriate use, and risks are recorded in the
[data card](docs/data-card.md); its dataset license is in
[`data/LICENSE.md`](data/LICENSE.md). The synthetic sample remains only as a
small deterministic test fixture.

## Quickstart from a Fresh Clone

Requirements: Python 3.11+.

```bash
git clone https://github.com/aemref/AI-01-production-ml-classifier.git
cd AI-01-production-ml-classifier
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/train.py
```

The last command trains and evaluates the baseline against the repository's
licensed real-data subset. No API key, external service, or data download is
required.

## Run

Run the complete baseline with one command:

```bash
python src/train.py
```

Expected output for the default real-data subset:

```text
Validation accuracy: 0.918
Validation F1: 0.933
Validation malignant recall: 0.906
Test accuracy: 0.860
Test F1: 0.882
Test malignant recall: 0.906
```

Input schema:

```csv
feature_a,feature_b,label
17.99,10.38,0
13.54,14.36,1
```

Here `feature_a` is mean radius, `feature_b` is mean texture, and labels `0` and
`1` mean malignant and benign respectively. Empty files, missing columns,
missing or non-finite values, non-numeric features, invalid labels, and data too
small for a stratified split produce clear validation errors. Training also
stops before splitting when a feature directly or inversely copies the binary
target, or when duplicate feature rows could cross partition boundaries.

## Test

```bash
python -m pytest -q
```

CI runs the same tests, baseline command, and data-quality audit on every push
and pull request.

## Data Quality Audit

Run the reproducible missingness, class-imbalance, and leakage-risk checks:

```bash
python src/data_quality.py
```

The current dataset has no missing values or duplicate feature rows. Its class
ratio is `1.684` (357 benign to 212 malignant), classified by the audit's
temporary heuristic as moderate imbalance. Neither feature copies the target,
and neither crosses the audit's temporary absolute-correlation threshold of
`0.95`. See the [risk analysis](docs/week-02-data-quality-risks.md) for findings,
limitations, and follow-up work.

The English [EDA notebook](notebooks/week-02-eda.ipynb) and its concise
[evaluation report](docs/week-02-eda-report.md) compare the original and
class-balanced baselines, including their error trade-off. The notebook is
committed with its outputs so the reported row counts, distributions, quality
signals, and model comparison can be inspected without rerunning it.

## Architecture

```mermaid
flowchart LR
    A[CSV input] --> B[Schema and value validation]
    B --> C[Stratified 70/15/15 split]
    C --> D[Train-only model fit]
    C --> E[Validation evaluation]
    C --> F[Untouched test evaluation]
    D --> E
    D --> F
    E --> G[Validation metrics]
    F --> H[Test metrics]
    I[pytest] --> J[GitHub Actions CI]
    B -. invalid input .-> K[Clear error]
```

## Repository Structure

```text
src/                         Source and model code
tests/                       Unit and regression tests
data/                        Small, safe development fixtures
docs/                        Measurements and engineering notes
notebooks/                   Reproducible English EDA notebooks
.github/workflows/           Continuous integration
.github/ISSUE_TEMPLATE/      Standard issue intake
```

## Metrics and Baseline

| Metric | Validation | Test |
| --- | ---: | ---: |
| Accuracy | 0.918 | 0.860 |
| Benign F1 | 0.933 | 0.882 |
| Malignant recall | 0.906 | 0.906 |

These deterministic values use only two of the source dataset's 30 features and
one 70/15/15 split. They are not a production or clinical performance claim.

## Limitations and Risks

- This historical dataset does not establish present-day clinical validity or
  represent deployment populations and drift.
- The baseline uses only mean radius and mean texture, discarding 28 source
  features for a deliberately small first integration.
- A single train/test split is insufficient for model selection.
- No model artifact, serving API, monitoring, or data versioning exists yet.
- A single malignant-recall metric still cannot replace a confusion matrix,
  uncertainty analysis, subgroup evaluation, or clinical validation.

## Security and Cost Notes

- Do not commit secrets, credentials, personal data, or `.env` files.
- Only reviewed, licensed, non-identifying data and synthetic fixtures belong
  in this repository.
- The baseline runs locally and uses no external API, cloud service, or paid
  compute; current runtime cost is effectively zero apart from local resources.
- Future production work must add access control, data retention rules, and a
  cost budget before using external infrastructure.

## Status

Week 2: Real classification data, dataset licensing, data card, stronger input
validation, tests, and CI integration.
