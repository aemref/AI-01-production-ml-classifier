# Week 1 Closeout

Date: 2026-08-31  
Scope: Baseline assessment, repository scaffold, documentation, and local integration.

## Completion evidence

| Evidence | Result |
| --- | --- |
| Integrated `main` commit before closeout | `d1add07` |
| Non-merge implementation commits | 6 |
| Unit test command | `3 passed in 0.69s` |
| Evaluation command | Accuracy `1.000`, F1 `1.000` |
| Local runtime benchmark | real `1.09s`, user `0.89s`, sys `0.15s` |
| Clean-clone test | Passed |
| Syntax/compile check | Passed |
| Working tree after verification | Clean |
| AWS usage/cost | N/A; no AWS integration exists |
| Application/submission count | N/A; no external application workflow exists |

## Demo

From the repository root:

```bash
python src/train.py --data data/sample.csv
```

Observed output:

```text
Accuracy: 1.000
F1: 1.000
```

This is the reproducible demo evidence for the current baseline. The fixture
is synthetic and intentionally small.

## Baseline assessment status

- Machine learning: baseline pipeline completed; Accuracy `1.000`, F1 `1.000`.
- Python: personal score still requires the learner's assessment.
- English: reading, writing, speaking, and vocabulary scores still require the learner's assessment.

## Retrospective

### What worked

- A small, deterministic CSV fixture made local and CI verification fast.
- Input validation and unit tests made failure behavior visible.
- Keeping the baseline command to one line made the project easy to run from a clean clone.
- Separating source, tests, data, docs, and GitHub configuration made the next steps clear.

### What should not be repeated

- Do not interpret a perfect score on a tiny synthetic fixture as production model quality.
- Do not ignore merge or CI state just because local tests pass.
- Do not leave personal baseline scores implicit; record the measurement method and date.
- Do not commit real credentials, personal data, or unreviewed datasets.

## Handoff to next week

The 12-week plan is recorded in `docs/12-week-issue-backlog.md`. The seven
next-week GitHub issues are intentionally not created in this closeout; they
remain an external tracking action.
