# Week 4: Production Inference Slice

Tracking scope: FastAPI inference, Pydantic request validation, explicit error
handling, a non-root Docker image, unit/integration coverage, CI smoke checks,
and measured latency and throughput. This slice is educational and is not
approved for clinical or production use.

## Interface contract

`POST /predict` accepts exactly two finite numeric fields:

```json
{"feature_a": 17.99, "feature_b": 10.38}
```

The response includes the predicted label, a human-readable class name, both
class probabilities, confidence, and a model version derived from the training
dataset checksum. `GET /health` confirms that the predictor loaded and exposes
the same model version.

The service loads the validation-selected, class-balanced logistic regression
from a portable JSON artifact at process startup. It verifies the SHA-256
sidecar and schema before serving traffic, and fails startup when either is
invalid. Building the artifact fits on the train partition only; it never fits
on validation or test rows. The [artifact and rollback policy](model-artifact.md)
records the format, trust boundary, release checks, and rollback procedure.

## Failure handling

Malformed JSON bodies, missing features, unknown fields, non-numeric values,
and non-finite numbers receive HTTP `422`. The response has a stable
`invalid_request` code, concise field-level details, and the same request ID as
the `X-Request-ID` response header.

Rejected requests emit a structured `request_rejected` warning with the request
ID, path, status, and error code. Request payloads are deliberately excluded to
avoid logging potentially sensitive input values. Integration tests cover the
successful, missing-field, unknown-field, and invalid-type paths.

## Container evidence

The image uses `python:3.11-slim`, runs as the unprivileged `app` user, includes
only runtime dependencies, and defines a `/health` check. On 2026-09-14 the
following checks completed against Docker Engine 29.5.2 on `linux/arm64`:

```text
docker build -t ai01-classifier:local .                  PASS
container /health request                               PASS
container /predict request                              PASS
runtime image excludes pytest and httpx                 PASS
```

CI repeats the Python 3.11 image build and both endpoint smoke checks on every
push and pull request.

## Measured baseline

Command:

```bash
python -m src.benchmark_api --requests 200 --warmup 20
```

Environment: local `arm64`, Python 3.14.7, one FastAPI `TestClient`, one request
at a time, model already loaded, 20 unmeasured warmups.

| Measure | Result |
|---|---:|
| Requests | 200 |
| Failed requests | 0 |
| Total measured time | 0.172 s |
| Throughput | 1,161.2 requests/s |
| Minimum latency | 0.810 ms |
| Median latency | 0.846 ms |
| p95 latency | 0.980 ms |
| Maximum latency | 1.772 ms |

These values measure the in-process HTTP application path, not network,
container scheduling, concurrency, TLS, or production load. They establish a
repeatable regression baseline only; they are not a capacity claim or service
level objective. CI runs a shorter 50-request sanity benchmark without asserting
machine-dependent timing thresholds.

## Reproduction

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m src.benchmark_api --requests 200 --warmup 20
docker build -t ai01-classifier .
docker run --rm -p 8000:8000 ai01-classifier
```

Next work: add concurrency and external-network container benchmarks before
defining any latency objective.
