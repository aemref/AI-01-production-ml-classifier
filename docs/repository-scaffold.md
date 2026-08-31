# Repository Scaffold

The repository is organized around a small, testable ML pipeline:

| Path | Responsibility |
| --- | --- |
| `src/` | Application and model code |
| `tests/` | Unit and regression tests |
| `data/` | Small fixtures and local datasets |
| `docs/` | Decisions, measurements, and experiment notes |
| `.github/workflows/` | Continuous integration |
| `.github/ISSUE_TEMPLATE/` | Standard issue intake |

## Validation signal

The baseline command is:

```bash
python src/train.py --data data/sample.csv
```

The expected output includes `Accuracy` and `F1`. CI runs the same command after the unit tests.
