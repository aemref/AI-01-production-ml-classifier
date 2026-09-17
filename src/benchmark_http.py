"""Benchmark the prediction API through an external HTTP connection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil
from typing import Sequence


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
