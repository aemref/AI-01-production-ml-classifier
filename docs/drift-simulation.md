# Controlled Feature-Shift Simulation

Run from the repository root after installing `requirements.txt`:

```bash
python -m src.drift --shift-sd 1
python -m src.drift --shift-sd 0
```

The command validates the checked-in dataset and its checksum-bound inference
artifact, uses the fixed stratified test partition (seed 42), and adds one test
partition standard deviation of mean radius (`feature_a`) to each row. Mean
texture remains unchanged. It then uses the same artifact for predictions
before and after the shift. `--shift-sd 0` is a negative control: it must
produce zero changed predictions. Invalid shifts and an artifact from another
dataset fail with a nonzero exit code.

## Measured local result

Run on 2026-09-20 with the checked-in dataset and model artifact, Python 3.11
environment, `--shift-sd 1`:

| Quantity | Observed |
| --- | ---: |
| Paired test rows | 86 |
| Mean radius before | 14.139895 |
| Mean radius after | 17.810841 |
| Added radius units | 3.670945 |
| Standardized mean radius change | 1.000000 |
| Mean texture change | 0 |
| Malignant predictions before | 37 |
| Malignant predictions after | 71 |
| Changed predictions | 34 |

The source dataset SHA-256 is
`98b12889accbae788456d9442fa0753d2d15a361241b543f9fe21795176d1303`;
the artifact version is `logistic-regression-98b12889accb`. These values
bind the report to the exact local inputs. The CLI emits the full JSON report,
including means and reference standard deviations, for independent inspection.

## Interpretation and limits

This is an imposed feature perturbation, not an observed production drift event.
The standardized mean change uses the unmodified test partition's population
standard deviation. A constant reference yields `null` for the standardized
value because it has no scale. The 34 changed predictions quantify model
sensitivity under this intervention; they are not errors, clinical outcomes, or
a performance degradation estimate. The shifted rows have no ground-truth
labels. No drift alarm threshold or deployment decision should be inferred from
this single, artificial scenario. Real monitoring would require representative
incoming data, a justified reference window, data governance, and calibrated
alert thresholds.
