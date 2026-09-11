# Week 3: Train/Validation/Test Pipeline

## Goal

Build a train/validation/test pipeline and add regression tests against split
leakage.

## Scope

In scope: a deterministic stratified 70/15/15 split, train-only model fitting,
separate validation/test metrics, disjointness and fit-boundary tests, updated
fixtures, input validation, CLI evidence, and documentation.

Out of scope: model comparison, validation-driven selection, hyperparameter
tuning, cross-validation, and a configurable split/seed API. The source contains
a visible `TODO` reserving validation-driven candidate selection for Week 4;
test data must remain untouched when that work begins.

## Test-first evidence

The leakage tests were added before `split_dataset` existed. Their first run
failed during collection as expected:

```text
ImportError: cannot import name 'split_dataset' from 'src.train'
1 error
```

The new tests require pairwise-disjoint partition indices, complete coverage of
the source rows, both labels in every partition, and proof that
`LogisticRegression.fit` receives train indices only.

### Content-leakage guard increment

Goal: improve the train/validation/test pipeline by blocking content-level
leakage before splitting.

Index-disjointness does not detect two records with different row indices but
identical feature content, nor does it detect a feature that encodes the target.
Two failure fixtures were therefore added before the guard implementation. The
focused run failed as expected:

```text
FAILED test_target_copy_failure_fixture_has_clear_leakage_error - Failed: DID NOT RAISE
FAILED test_duplicate_feature_failure_fixture_has_clear_leakage_error - Failed: DID NOT RAISE
2 failed, 16 passed
```

The training entry point now rejects direct and inverse binary target copies and
duplicate feature rows before any partition or model fit is created. The latter
is deliberately conservative. A visible source `TODO` requires replacement
with group-aware splitting after the data contract gains a stable patient or
observation-group identifier.

## Design and leakage controls

- The first stratified split reserves 30% as holdout data.
- A second stratified split divides the holdout equally into validation and test.
- Original DataFrame indices are preserved so overlap can be tested directly.
- Logistic Regression is fitted once, using only the 398-row train partition.
- Validation is reported independently on 85 rows.
- Test is reported independently on 86 rows and is not used for fitting.
- No learned preprocessing currently exists. Any future transformer must be fit
  on train only and protected by an equivalent regression test.
- Content checks reject direct/inverse target copies and duplicate feature rows
  before splitting, covering leakage that index-overlap assertions cannot see.

## Empty, invalid, and boundary behavior

- Header-only CSV input is rejected as empty.
- Missing columns, missing/non-numeric/non-finite values, and invalid labels
  retain their clear errors.
- Validation/test fractions must each be positive and sum to less than one.
- Data too small to place both classes in all three partitions receives a
  pipeline-specific error.
- `tests/fixtures/boundary_minimum_split.csv` contains the minimum balanced
  12-row boundary supported by the default 70/15/15 split.

## Verification

```bash
.venv/bin/python -m pytest -q
.venv/bin/python src/train.py
.venv/bin/python src/data_quality.py
.venv/bin/python -m compileall -q src tests
```

Final verification result: `24 passed in 0.79s`; training, data-quality audit,
and byte-compilation commands all exited with status `0`. The audit again
reported 569 rows, 0 missing values, 0 duplicate feature rows, and no target-copy
features.

Observed training output:

```text
Validation accuracy: 0.918
Validation F1: 0.933
Validation malignant recall: 0.906
Test accuracy: 0.860
Test F1: 0.882
Test malignant recall: 0.906
```

The metric change from Week 2 is expected because the fixed test membership is
now defined by a separate 15% partition rather than the earlier 25% holdout.
