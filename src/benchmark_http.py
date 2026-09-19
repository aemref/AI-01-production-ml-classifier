"""Benchmark the prediction API through an external HTTP connection."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from math import ceil, isfinite
import socket
from time import perf_counter
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


DEFAULT_PAYLOAD = {"feature_a": 17.99, "feature_b": 10.38}
MAX_CONCURRENCY = 64


@dataclass(frozen=True)
class RequestObservation:
    """One completed HTTP request, including transport failures."""

    duration_seconds: float
    status_code: int | None
    failure_type: str | None = None


@dataclass(frozen=True)
class HttpBenchmarkResult:
    """Machine-readable external benchmark summary."""

    request_count: int
    success_count: int
    failure_count: int
    total_wall_seconds: float
    throughput_requests_per_second: float
    latency_ms_min: float
    latency_ms_median: float
    latency_ms_p95: float
    latency_ms_max: float
    status_counts: dict[str, int]
    failure_types: dict[str, int]


def normalize_base_url(base_url: str) -> str:
    """Validate a benchmark target and return it without a trailing slash."""
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Base URL must be an absolute HTTP or HTTPS URL")
    if parsed.username or parsed.password:
        raise ValueError("Base URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("Base URL must not contain a query or fragment")

    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _valid_prediction_response(body: bytes) -> bool:
    """Reject unrelated HTTP 200 responses and malformed classifier results."""
    try:
        prediction = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(prediction, dict):
        return False

    label = prediction.get("label")
    if type(label) is not int or label not in (0, 1):
        return False
    if prediction.get("label_name") != {0: "malignant", 1: "benign"}[label]:
        return False
    if not isinstance(prediction.get("model_version"), str) or not prediction["model_version"]:
        return False

    probabilities = [
        prediction.get("malignant_probability"),
        prediction.get("benign_probability"),
    ]
    confidence = prediction.get("confidence")
    if any(
        type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1
        for value in [*probabilities, confidence]
    ):
        return False
    if abs(sum(probabilities) - 1) > 1e-6:
        return False
    return abs(confidence - probabilities[label]) <= 1e-6


def perform_prediction_request(
    base_url: str,
    *,
    timeout_seconds: float,
    clock=perf_counter,
    opener=urlopen,
) -> RequestObservation:
    """Send one prediction request and retain failures as observations."""
    if timeout_seconds <= 0:
        raise ValueError("Request timeout must be positive")

    target = f"{normalize_base_url(base_url)}/predict"
    request = Request(
        target,
        data=json.dumps(DEFAULT_PAYLOAD).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_at = clock()
    try:
        with opener(request, timeout=timeout_seconds) as response:
            body = response.read()
            status_code = response.status
        failure_type = (
            "invalid_response"
            if status_code == 200 and not _valid_prediction_response(body)
            else None
        )
        return RequestObservation(clock() - started_at, status_code, failure_type)
    except HTTPError as error:
        return RequestObservation(clock() - started_at, error.code)
    except (TimeoutError, socket.timeout):
        return RequestObservation(clock() - started_at, None, "timeout")
    except URLError as error:
        failure_type = (
            "timeout"
            if isinstance(error.reason, (TimeoutError, socket.timeout))
            else "connection_error"
        )
        return RequestObservation(clock() - started_at, None, failure_type)


def run_http_benchmark(
    base_url: str,
    *,
    request_count: int = 200,
    warmup_count: int = 20,
    concurrency: int = 4,
    timeout_seconds: float = 2.0,
    requester: Callable[..., RequestObservation] = perform_prediction_request,
    clock=perf_counter,
) -> HttpBenchmarkResult:
    """Run bounded concurrent requests against an already-started service."""
    if request_count <= 0 or warmup_count < 0:
        raise ValueError("Request count must be positive and warmup cannot be negative")
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValueError(f"Concurrency must be between 1 and {MAX_CONCURRENCY}")
    if timeout_seconds <= 0:
        raise ValueError("Request timeout must be positive")

    normalized_url = normalize_base_url(base_url)
    for _ in range(warmup_count):
        observation = requester(
            normalized_url,
            timeout_seconds=timeout_seconds,
        )
        if observation.status_code != 200 or observation.failure_type:
            failure = observation.failure_type or f"http_{observation.status_code}"
            raise RuntimeError(f"Warmup request failed: {failure}")

    started_at = clock()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                requester,
                normalized_url,
                timeout_seconds=timeout_seconds,
            )
            for _ in range(request_count)
        ]
        observations = [future.result() for future in futures]
    wall_seconds = clock() - started_at

    return summarize_observations(observations, wall_seconds=wall_seconds)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a running inference API over HTTP",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Service root URL; /predict is appended automatically",
    )
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args(argv)

    try:
        target = normalize_base_url(args.base_url)
        result = run_http_benchmark(
            target,
            request_count=args.requests,
            warmup_count=args.warmup,
            concurrency=args.concurrency,
            timeout_seconds=args.timeout,
        )
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))

    report = {
        "target": target,
        "warmup_count": args.warmup,
        "concurrency": args.concurrency,
        "timeout_seconds": args.timeout,
        **asdict(result),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = max(1, ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def summarize_observations(
    observations: Sequence[RequestObservation],
    *,
    wall_seconds: float,
) -> HttpBenchmarkResult:
    """Summarize attempts using wall time so concurrent throughput is honest."""
    if not observations:
        raise ValueError("At least one measured request is required")
    if wall_seconds <= 0:
        raise ValueError("Wall time must be positive")
    if any(item.duration_seconds < 0 for item in observations):
        raise ValueError("Request durations cannot be negative")

    durations = sorted(item.duration_seconds for item in observations)
    midpoint = len(durations) // 2
    median = (
        durations[midpoint]
        if len(durations) % 2
        else (durations[midpoint - 1] + durations[midpoint]) / 2
    )
    status_counts = Counter(
        str(item.status_code) if item.status_code is not None else "transport_error"
        for item in observations
    )
    failure_types = Counter(
        item.failure_type or f"http_{item.status_code}"
        for item in observations
        if item.status_code != 200 or item.failure_type
    )
    success_count = sum(
        item.status_code == 200 and item.failure_type is None
        for item in observations
    )

    return HttpBenchmarkResult(
        request_count=len(observations),
        success_count=success_count,
        failure_count=len(observations) - success_count,
        total_wall_seconds=wall_seconds,
        throughput_requests_per_second=len(observations) / wall_seconds,
        latency_ms_min=durations[0] * 1000,
        latency_ms_median=median * 1000,
        latency_ms_p95=_nearest_rank(durations, 0.95) * 1000,
        latency_ms_max=durations[-1] * 1000,
        status_counts=dict(sorted(status_counts.items())),
        failure_types=dict(sorted(failure_types.items())),
    )
