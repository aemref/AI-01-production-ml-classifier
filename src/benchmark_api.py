"""Measure repeatable in-process HTTP inference latency and throughput."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import json
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi.testclient import TestClient

from src.api import DEFAULT_DATA_PATH, create_app
from src.predictor import Predictor


DEFAULT_PAYLOAD = {"feature_a": 17.99, "feature_b": 10.38}


@dataclass(frozen=True)
class BenchmarkResult:
    request_count: int
    failure_count: int
    total_seconds: float
    throughput_requests_per_second: float
    latency_ms_min: float
    latency_ms_median: float
    latency_ms_p95: float
    latency_ms_max: float


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = max(1, ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def summarize_durations(
    durations: Sequence[float],
    *,
    failure_count: int,
) -> BenchmarkResult:
    """Summarize request durations measured in seconds."""
    if not durations:
        raise ValueError("At least one measured request is required")

    total_seconds = sum(durations)
    ordered = sorted(durations)
    midpoint = len(ordered) // 2
    median = (
        ordered[midpoint]
        if len(ordered) % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2
    )
    return BenchmarkResult(
        request_count=len(durations),
        failure_count=failure_count,
        total_seconds=total_seconds,
        throughput_requests_per_second=len(durations) / total_seconds,
        latency_ms_min=ordered[0] * 1000,
        latency_ms_median=median * 1000,
        latency_ms_p95=_nearest_rank(ordered, 0.95) * 1000,
        latency_ms_max=ordered[-1] * 1000,
    )


def benchmark_client(
    client: Any,
    *,
    request_count: int,
    warmup_count: int,
    clock: Callable[[], float] = perf_counter,
) -> BenchmarkResult:
    """Benchmark a client exposing ``post`` while keeping warmups unmeasured."""
    if request_count <= 0 or warmup_count < 0:
        raise ValueError("Request count must be positive and warmup cannot be negative")

    for _ in range(warmup_count):
        client.post("/predict", json=DEFAULT_PAYLOAD)

    durations = []
    failure_count = 0
    for _ in range(request_count):
        started_at = clock()
        response = client.post("/predict", json=DEFAULT_PAYLOAD)
        durations.append(clock() - started_at)
        if response.status_code != 200:
            failure_count += 1

    return summarize_durations(durations, failure_count=failure_count)


def run_benchmark(
    data_path: str | Path = DEFAULT_DATA_PATH,
    *,
    request_count: int = 200,
    warmup_count: int = 20,
) -> BenchmarkResult:
    predictor = Predictor.from_dataset(data_path)
    with TestClient(create_app(predictor)) as client:
        return benchmark_client(
            client,
            request_count=request_count,
            warmup_count=warmup_count,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the inference API")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    try:
        result = run_benchmark(
            args.data,
            request_count=args.requests,
            warmup_count=args.warmup,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
