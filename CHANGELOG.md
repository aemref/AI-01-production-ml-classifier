# Changelog

All notable changes to this project are documented here. The project follows
Semantic Versioning for the application interface; the dataset-derived model
artifact keeps its separate immutable version.

## 1.0.0 — 2026-09-22

### Added

- Licensed and documented Breast Cancer Wisconsin data subset with structural
  quality and leakage-risk checks.
- Leakage-safe stratified train, validation, and test partitions plus a
  validation-only comparison of Logistic Regression, Decision Tree, and Random
  Forest candidates.
- Model/data cards, error analysis, deterministic drift simulation, portable
  experiment reports, and opt-in local MLflow tracking.
- Checksummed, schema-validated JSON model artifact with a tested rollback
  policy and inference implementation independent of pickled model objects.
- Typed FastAPI health and prediction endpoints, structured validation errors,
  request correlation, non-root Docker image, and local HTTP benchmarks.
- Reproducible JSON demo, audited demo animation, architecture document, and
  CI coverage for tests, commands, artifact verification, container smoke, and
  bounded benchmark paths.

### Safety and limitations

- This educational two-feature model is not clinically validated and must not
  be used for diagnosis, triage, treatment, or patient decisions.
- Local latency and throughput measurements are regression evidence, not a
  capacity claim or service-level objective.
- The service has no authentication, rate limiting, signed provenance,
  persistent telemetry, or real-world drift monitor.
