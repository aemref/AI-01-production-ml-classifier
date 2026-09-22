# v1.0.0 Release Checklist

The application version and model version are deliberately separate:
`1.0.0` describes the public repository/API contract, while
`logistic-regression-98b12889accb` identifies model parameters built from the
complete dataset digest. A release tag is created only after every gate below
passes from a clean `main` worktree.

## Source and evidence gates

- [x] Licensed dataset, provenance, transformations, and intended use recorded.
- [x] Train, validation, and test roles are distinct; selection does not inspect
  test metrics.
- [x] At least three fixed model candidates are compared on precision, recall,
  F1, and ROC-AUC.
- [x] Model card, data card, architecture, drift study, experiment evidence,
  failure taxonomy, security limits, cost notes, and rollback policy published.
- [x] JSON artifact and SHA-256 sidecar are checked in and fail closed on
  integrity, schema, feature, label, or dataset mismatch.
- [x] English README contains clean-clone setup, API use, measurement limits,
  architecture, reproducible demo, and explicit non-clinical warnings.
- [x] Demo animation is generated from checked-in SVG frames after the real demo
  command succeeds; no external result or user feedback is implied.

## Final verification commands

Run in this order before creating `v1.0.0`:

```bash
python -m pytest -q
python src/train.py --data data/breast_cancer_wisconsin_diagnostic.csv
python -m src.compare_models --data data/breast_cancer_wisconsin_diagnostic.csv
python -m src.run_experiment --output /tmp/ai01-model-comparison.json
python src/data_quality.py --data data/breast_cancer_wisconsin_diagnostic.csv
python -m src.drift --shift-sd 1
python -m src.drift --shift-sd 0
python -m src.verify_artifact
python -m src.demo --compact
python -m compileall -q src tests
python -m src.benchmark_api --requests 50 --warmup 5
docker build --tag ai01-classifier:release .
```

Then run the container as a non-root user, verify `/health` and `/predict`, and
execute the 50-request external benchmark with four workers and a two-second
timeout. Finish with `git diff --check`, confirm the worktree is clean, and
verify the pushed commit and annotated tag resolve from `origin/main`.

## Release policy

Do not tag or push if any command fails, Git history has diverged, the working
tree contains unexplained files, the container does not become healthy, or the
remote main branch cannot be verified. Never rebase, force-push, rewrite the
tag, or weaken a gate to make the release pass.
