# Week 3 Issue Plan: Reproducible Training Configuration

Each item is deliberately scoped to one focused session. Publish these bodies
as separate GitHub issues once repository write authentication is restored.

## Issue 1 — Make the training seed configurable

### Input

`src/train.py` currently hard-codes random seed `42` in the split and model.
Preserve `42` as the default so the current baseline remains reproducible.

### Acceptance criteria

- `train_and_evaluate` accepts a `random_state` integer and uses it for both the
  stratified split and Logistic Regression.
- The CLI accepts `--random-state`; omission retains default `42`.
- Seed `42` preserves Accuracy `0.888`, benign F1 `0.908`, and malignant recall
  `0.906` on the checked-in real dataset.
- A regression test proves two calls with the same seed return the same metrics.
- README usage documents the option without making a production claim.

### Run command

```bash
python -m pytest tests/test_train.py -q
python src/train.py --random-state 42
```

## Issue 2 — Validate a configurable test split

Add a `test_size` argument and `--test-size` CLI option, retain `0.25` as the
default, reject values outside `(0, 1)`, and test one valid boundary plus invalid
zero/one inputs.

Validation command: `python -m pytest tests/test_train.py -q`.

## Issue 3 — Expose the class-weight strategy

Allow `balanced` and `none` as explicit model settings, keep `balanced` as the
default, reject unknown values, and test both supported paths without changing
the real-data regression thresholds.

Validation command: `python src/train.py --class-weight balanced`.

## Issue 4 — Emit a machine-readable experiment manifest

Add an opt-in JSON output containing dataset path/checksum, seed, test size,
class-weight strategy, package version, and metrics. Keep the existing text CLI
output unchanged by default.

Validation command: `python src/train.py --output-format json`.

## Issue 5 — Add a repeated-run determinism regression

Run the same checked-in dataset/configuration twice in one integration test and
assert identical metrics. Document that this protects deterministic local/CI
runs, not cross-platform floating-point identity for every environment.

Validation command: `python -m pytest -q -k determinism`.

## Issue 6 — Test the configured CLI end to end

Add subprocess integration coverage for valid configuration, a malformed seed,
and an invalid split. Assert exit codes and concise error messages rather than
internal implementation details.

Validation command: `python -m pytest -q -k cli`.

## Issue 7 — Add a Ruff lint gate to CI

Add a bounded Ruff development dependency and minimal configuration, make the
current `src/` and `tests/` tree pass, document the local command, and add the
same command to GitHub Actions.

Validation command: `python -m ruff check src tests`.

## Publication blocker

At closeout, GitHub issue creation could not be performed because `gh auth
status` reported an invalid token and the available browser session was signed
out. This does not block repository code or documentation. After re-authentication,
create seven issues from the sections above, close stale completed Issue #1, and
replace this note with the resulting issue links.
