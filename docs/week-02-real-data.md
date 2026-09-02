# Week 2: Real Classification Data

## Goal

Select a real classification dataset and add its license and data card.

## Scope

In scope: one repository-safe real dataset subset, provenance and license
documentation, the smallest working main-flow integration, clear validation for
empty/invalid/boundary inputs, regression tests, and CI coverage.

Out of scope: full 30-feature modeling, hyperparameter tuning, clinical claims,
deployment, dashboards, exhaustive EDA, and model selection.

## Test-first evidence

The real-data regression test was added before the dataset. Its first run failed
as expected with:

```text
FileNotFoundError: Dataset not found: .../data/breast_cancer_wisconsin_diagnostic.csv
1 failed
```

This established the expected behavior: the real dataset must pass through the
existing `train_and_evaluate` entry point and produce Accuracy and F1 of at least
`0.85` under the fixed split.

## Reproduction

```bash
.venv/bin/python -m pytest -q
.venv/bin/python src/train.py
```

Observed results:

```text
7 passed in 1.08s
Accuracy: 0.888
F1: 0.912
```
