"""Benchmark the prediction API through an external HTTP connection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from math import ceil
import socket
from time import perf_counter
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


DEFAULT_PAYLOAD = {"feature_a": 17.99, "feature_b": 10.38}


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
            response.read()
            status_code = response.status
        return RequestObservation(clock() - started_at, status_code)
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
        if item.status_code != 200
    )
    success_count = sum(item.status_code == 200 for item in observations)

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
