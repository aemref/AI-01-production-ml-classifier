# Week 2 Closeout

Date: 2026-09-07
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
| GitHub Actions on closeout commit | [`success`, run `34031471112`](https://github.com/aemref/AI-01-production-ml-classifier/actions/runs/34031471112) |
| Working tree after verification | Clean |
| Clean-clone rerun | `15 passed in 0.69s`; compile check passed |
| Clean-clone runtime benchmark | training command: real `0.74s` |
| Issue evidence comment | [Issue #3 comment](https://github.com/aemref/AI-01-production-ml-classifier/issues/3#issuecomment-5559030184) |

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

## Open issue classification

| Item | Classification | Reason and action |
| --- | --- | --- |
| [Issue #1](https://github.com/aemref/AI-01-production-ml-classifier/issues/1) | Completed; administrative follow-up | The baseline, repository setup, CI, and documentation exist on `main`. Close the stale issue when GitHub CLI/web write access is restored. |
| [Issue #3](https://github.com/aemref/AI-01-production-ml-classifier/issues/3) | Completed and closed | Week 2 evidence and the successful CI link were posted before closure. |
| Week 3 configuration work | Deferred by schedule | Split into seven one-session issues in `docs/week-03-issue-plan.md`; no Week 2 scope is silently carried forward. |
| Publish the seven Week 3 GitHub issues | Process blocker | Issue bodies are ready, but the local GitHub CLI token is invalid and the available browser session is signed out. Code, tests, and `main` integration are not blocked. |
| Python lint gate | Deferred follow-up | No linter is configured today; syntax compilation passes. Issue 7 adds Ruff explicitly instead of reporting an unconfigured lint as successful. |

## Weekly progress counts

| Progress signal | Count | Evidence |
| --- | ---: | --- |
| Non-merge Week 2 commits | 7 | Includes the final handoff/issue-plan commit. |
| Tests | 15 passing | Clean clone: `python -m pytest -q`. |
| Benchmark runs recorded | 1 | Clean-clone training runtime: real `0.74s`. |
| Reproducible demos | 1 | `python src/train.py` on the licensed real-data subset. |
| English writing artifacts | 6 | Dataset license, data card, real-data log, risk report, EDA report, and EDA notebook; README also updated. |
| External issue/PR reviews | 1 | `scikit-learn/scikit-learn#16479`, read-only. |
| Applications submitted | 0 | No application workflow was in Week 2 scope. |
| AWS progress events | 0 | The project has no AWS integration or spend. |

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
- Do not leave stale completed issues open or depend on an expired CLI login at
  closeout; verify GitHub write access earlier in the week.

## Handoff to next week

Issue [#3](https://github.com/aemref/AI-01-production-ml-classifier/issues/3)
is closed with the evidence above. The seven Week 3 tasks and ready-to-run first
issue are in `docs/week-03-issue-plan.md`; the 12-week plan remains the roadmap
source of truth.
