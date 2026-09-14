from types import SimpleNamespace

import pytest

from src.benchmark_api import benchmark_client, summarize_durations


class StubClient:
    def __init__(self, statuses):
        self.statuses = iter(statuses)
        self.calls = []

    def post(self, path, *, json):
        self.calls.append((path, json))
        return SimpleNamespace(status_code=next(self.statuses))


def test_duration_summary_uses_nearest_rank_p95():
    result = summarize_durations(
        [0.001, 0.002, 0.003, 0.004, 0.005],
        failure_count=1,
    )

    assert result.request_count == 5
    assert result.failure_count == 1
    assert result.total_seconds == pytest.approx(0.015)
    assert result.throughput_requests_per_second == pytest.approx(5 / 0.015)
    assert result.latency_ms_median == pytest.approx(3.0)
    assert result.latency_ms_p95 == pytest.approx(5.0)


def test_benchmark_excludes_warmup_and_counts_failures():
    client = StubClient([200, 200, 422, 200])
    timestamps = iter([0.0, 0.010, 1.0, 1.020, 2.0, 2.030])

    result = benchmark_client(
        client,
        request_count=3,
        warmup_count=1,
        clock=lambda: next(timestamps),
    )

    assert len(client.calls) == 4
    assert result.request_count == 3
    assert result.failure_count == 1
    assert result.latency_ms_min == pytest.approx(10.0)
    assert result.latency_ms_median == pytest.approx(20.0)
    assert result.latency_ms_p95 == pytest.approx(30.0)


@pytest.mark.parametrize(
    ("request_count", "warmup_count"),
    [(0, 0), (-1, 0), (1, -1)],
)
def test_benchmark_rejects_invalid_counts(request_count, warmup_count):
    with pytest.raises(ValueError, match="positive"):
        benchmark_client(
            StubClient([]),
            request_count=request_count,
            warmup_count=warmup_count,
        )
