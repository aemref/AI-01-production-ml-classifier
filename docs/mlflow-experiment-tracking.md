# Local MLflow Experiment Tracking

## Purpose

The project can record its fixed three-model comparison in a local MLflow
backend. Tracking is opt-in: normal training, testing, artifact verification,
and API serving do not import MLflow or create tracking files.

Each run starts from the same validated dataset and train/validation/test split
used by `src.compare_models`. The command first writes a portable JSON record,
then logs that record and its selected fields to MLflow. This ordering keeps the
auditable input to the tracking adapter available even if logging fails.

## Reproduce Locally

```bash
python -m pip install -r requirements-mlflow.txt
python -m src.run_experiment \
  --track-mlflow \
  --run-name local-baseline
```

The default backend is the local SQLite database `mlruns.db`. Run artifacts are
stored below `mlruns/`, and the portable report is written to
`reports/model-comparison.json`. All three paths are ignored by Git because they
contain machine-local experiment state. Override `--tracking-uri`,
`--experiment-name`, or `--output` when a different local layout is required.

The tracked contract includes:

- dataset filename, row count, and SHA-256 digest;
- train, validation, and test row counts;
- the validation-only selection policy and selected candidate;
- four validation metrics for each of three candidates;
- four test metrics for the selected candidate; and
- the complete JSON comparison record as an MLflow artifact.

## Verified Run

A local run was executed on 2026-09-21 with Python 3.14.7 and MLflow 3.16.1.
MLflow reported run `c26e0096147e4b318f931e2855d95798` as `FINISHED`, with
8 parameters, 16 metrics, and the JSON report artifact.

| Field | Observed value |
| --- | --- |
| Dataset rows | 569 |
| Dataset SHA-256 | `98b12889accbae788456d9442fa0753d2d15a361241b543f9fe21795176d1303` |
| Split rows | 398 train / 85 validation / 86 test |
| Selected model | Logistic Regression |
| Test malignant precision | 0.784 |
| Test malignant recall | 0.906 |
| Test malignant F1 | 0.841 |
| Test ROC-AUC | 0.948 |

These figures reproduce the existing deterministic comparison; MLflow records
them but does not make them stronger evidence. The run uses one historical
dataset, two features, and one fixed split. It is not a clinical validation,
real-world drift measurement, cross-validation result, or deployment monitor.
The local database is not a shared or durable tracking service, and this flow
does not log a serialized model because the published inference artifact has a
separate checksum and release contract.
