# Week 1 Integration Checklist

## Scope

Integrate the baseline classifier, repository scaffold, CI, issue templates,
and documentation into `main`.

## Verification commands

Run from a clean clone after installing `requirements.txt`:

```bash
python -m pytest -q
python src/train.py --data data/sample.csv
git diff --check
```

Expected signals:

- Unit tests pass.
- The evaluation command reports Accuracy and F1.
- The working tree has no unintended changes.
- GitHub Actions completes successfully for the integrated commit.

## Current baseline evidence

| Signal | Result |
| --- | --- |
| Unit tests | 3 passed |
| Accuracy | 1.000 |
| F1 | 1.000 |
| Input validation | Empty, missing-column, missing-value, and single-class checks |

The metrics come from the small synthetic fixture and are not a production
quality claim.

## Integration result

- [x] Baseline branch merged into `main`
- [x] Repository scaffold merged into `main`
- [x] Documentation branch merged into `main`
- [x] Clean-clone verification completed
- [ ] CI result linked in the Week 1 issue
- [ ] Remaining work converted into tracked issues
