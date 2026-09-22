# v1.0.0 Release Evidence

Verification date: 2026-09-22 (Europe/Istanbul)  
Verified source commit: `3ea5f14`  
Application version: `1.0.0`  
Model version: `logistic-regression-98b12889accb`

This record contains only locally executed results. It does not claim external
users, deployment, clinical performance, or production capacity.

## Source and model verification

| Gate | Measured result |
| --- | --- |
| Full test suite | 123 passed; one upstream AnyIO deprecation warning |
| Training regression | validation accuracy 0.918; test accuracy 0.860 |
| Three-model comparison | Logistic Regression selected on validation |
| Selected-model test metrics | precision 0.784; recall 0.906; F1 0.841; ROC-AUC 0.948 |
| Dataset audit | 569 rows; no missing values, target copies, or duplicate feature rows |
| Artifact verification | dataset SHA-256 `98b12889accbae788456d9442fa0753d2d15a361241b543f9fe21795176d1303` matched |
| Drift simulation | shift 1 changed 34/86 predictions; zero-shift control changed 0/86 |
| Release evidence command | all four deterministic checks passed |
| Compile and diff checks | passed |

No standalone lint command is configured. Python compilation and the complete
test suite are the repository's current static and behavioral gates.

## Measured API paths

The in-process 50-request sanity run used five warmups and reported 50 successes,
zero failures, 1,326.2 requests/s, and 0.879 ms p95 latency on the local host.

The release Docker image was built from `python:3.11-slim`, became healthy, ran
as the unprivileged `app` user, and returned the expected model version from
both `/health` and `/predict`. A separate host process then measured the running
container over loopback with five warmups, 50 requests, four workers, and a
two-second timeout:

| Measure | Result |
| --- | ---: |
| Successful / failed requests | 50 / 0 |
| Throughput | 1,043.1 requests/s |
| Median latency | 3.576 ms |
| p95 latency | 5.508 ms |
| Maximum latency | 6.125 ms |

These single-machine measurements are regression evidence only. They are not a
service-level objective or a production capacity claim.

## Release decision

The source, test, documentation, artifact, demo, and container gates required by
the release checklist passed. The repository is eligible for the annotated
`v1.0.0` tag after this evidence record itself passes tests on a clean main
worktree and the remote branch is rechecked for divergence.
