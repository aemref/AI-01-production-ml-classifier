# Week 2 Closeout

Date: 2026-09-06
Scope: Data quality audit, real-dataset EDA, malignant-recall regression fix, and documentation reproducibility.

## Completion evidence

| Evidence | Result |
| --- | --- |
| Integrated `main` commit before closeout | `15be3d8` |
| Non-merge implementation commits since Week 1 | 5 |
| Unit test command | `python -m pytest -q` -> `15 passed in 30.5s` |
| Evaluation command | `python src/train.py` -> Accuracy `0.888`, F1 `0.908`, malignant recall `0.906` |
| Data quality command | `python src/data_quality.py` -> 0 missing, 0 duplicates, 0 target-copy features |
| Clean-clone test | Passed (`/tmp/ai01-clean`, fresh venv, `main` branch) |
| GitHub Actions on integrated commit | `success` (run `33960110431`) |
| Working tree after verification | Clean |

## Closing criteria checklist

- [x] EDA report published: `docs/week-02-eda-report.md`
- [x] Data card published and updated for the real dataset: `docs/data-card.md`
- [x] Data quality and leakage risks documented: `docs/week-02-data-quality-risks.md`
- [x] One external repository issue/PR reviewed (read-only, no comment posted)

## External review

Reviewed `scikit-learn/scikit-learn` issue
[#16479](https://github.com/scikit-learn/scikit-learn/issues/16479),
"Decision Tree probabilities with balanced class weight" — directly relevant
because this project also trains with `class_weight="balanced"` on the same
`load_breast_cancer` source dataset.

Summary: the reporter expected `predict_proba` to reflect pre-balancing class
rates but observed post-balancing probabilities. Maintainers confirmed the
current behavior is correct (`class_weight="balanced"` is equivalent to
reweighting/oversampling, so leaf probabilities reflect the balanced sample by
design) and the discussion moved to whether a parameter should expose
pre-balancing probabilities, tracked separately for API consistency with
sample-weight invariance. No comment was posted; this was a read-only review.

Takeaway for this project: our reported `malignant recall` is computed from
`predict`, not from raw `predict_proba`, so this project is not exposed to the
ambiguity discussed in the issue. Worth remembering if the project later
starts consuming `predict_proba` directly for anything threshold-sensitive.

## Retrospective

### What worked

- Test-first data quality checks caught the fixture-vs-real-data gap before
  it reached `main`.
- Adding malignant recall as an explicit metric surfaced a 9-of-89 miss rate
  that accuracy and benign F1 alone did not reveal.
- The clean-clone verification step continues to catch reproducibility drift
  before closeout.

### What should not be repeated

- Do not treat a single correlation/outlier heuristic as a leakage proof;
  it is a screening signal, not a verdict.
- Do not defer the external-review practice to the end of the week; it is
  easy to run out of time for it once implementation work is done.

## Handoff to next week

Issue [#3](https://github.com/aemref/AI-01-production-ml-classifier/issues/3)
acceptance criteria are satisfied by the evidence above; it can be closed.
The 12-week plan in `docs/12-week-issue-backlog.md` remains the source of
truth for what starts next.
