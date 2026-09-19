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

## External HTTP measurement

On 2026-09-19, the API ran under Uvicorn on local loopback (`127.0.0.1:8001`)
with Python 3.14.7 on `arm64`. The separate benchmark process sent 20 warmups
and then 200 measured prediction requests with four concurrent workers and a
two-second per-request timeout:

```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8001
python -m src.benchmark_http --base-url http://127.0.0.1:8001 \
  --requests 200 --warmup 20 --concurrency 4 --timeout 2
```

| Measure | Result |
|---|---:|
| Successful / failed requests | 200 / 0 |
| Total measured wall time | 0.119 s |
| Throughput | 1,673.7 requests/s |
| Minimum latency | 1.278 ms |
| Median latency | 2.305 ms |
| p95 latency | 2.667 ms |
| Maximum latency | 3.495 ms |

The benchmark validates the prediction response, counts HTTP errors, transport
errors, timeouts, and invalid `200` bodies separately, and exits nonzero when
measured requests fail. It prints JSON even on measured failure so CI retains
the failure breakdown. Throughput uses wall time; latency includes each full
HTTP request and response. This is a local-process TCP measurement, not a Docker
or production measurement. No timing threshold is asserted across machines.

## Container HTTP measurement

On the same date, Docker Engine 29.5.2 built the image from the repository's
Dockerfile for `linux/arm64` with Python 3.11.16 inside the container. The
container reported `healthy`, served `/health` and `/predict` from the host,
ran as `app`, and contained neither `pytest` nor `httpx`. The host benchmark
process then used the same 200 requests, 20 warmups, four workers, and two-second
timeout against the published loopback port:

| Measure | Result |
|---|---:|
| Successful / failed requests | 200 / 0 |
| Total measured wall time | 0.166 s |
| Throughput | 1,203.8 requests/s |
| Minimum latency | 1.314 ms |
| Median latency | 3.117 ms |
| p95 latency | 4.870 ms |
| Maximum latency | 7.337 ms |

This includes host-to-container HTTP and local container scheduling. The
single-machine run is a reproducible baseline, not a production load test or
capacity guarantee. CI repeats a shorter 50-request container benchmark and
fails if any measured request fails; it does not gate on host-specific timing.

## Reproduction

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m src.benchmark_api --requests 200 --warmup 20
docker build -t ai01-classifier .
docker run --rm -p 8000:8000 ai01-classifier
```

Start the API or container in another terminal, then run the external HTTP
benchmark command above against its listening port. CI runs that command against
the container after its smoke test.
