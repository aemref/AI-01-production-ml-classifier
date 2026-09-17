import pytest

from src.benchmark_http import RequestObservation, summarize_observations


def test_external_summary_uses_wall_time_and_preserves_failures():
    result = summarize_observations(
        [
            RequestObservation(0.010, 200),
            RequestObservation(0.020, 422),
            RequestObservation(0.030, None, "connection_error"),
            RequestObservation(0.040, 200),
        ],
        wall_seconds=0.050,
    )

    assert result.request_count == 4
    assert result.success_count == 2
    assert result.failure_count == 2
    assert result.throughput_requests_per_second == pytest.approx(80.0)
    assert result.latency_ms_median == pytest.approx(25.0)
    assert result.latency_ms_p95 == pytest.approx(40.0)
    assert result.status_counts == {"200": 2, "422": 1, "transport_error": 1}
    assert result.failure_types == {"connection_error": 1, "http_422": 1}


@pytest.mark.parametrize(
    ("observations", "wall_seconds", "message"),
    [
        ([], 1.0, "At least one"),
        ([RequestObservation(0.1, 200)], 0.0, "Wall time"),
        ([RequestObservation(-0.1, 200)], 1.0, "cannot be negative"),
    ],
)
def test_external_summary_rejects_invalid_measurements(
    observations,
    wall_seconds,
    message,
):
    with pytest.raises(ValueError, match=message):
        summarize_observations(observations, wall_seconds=wall_seconds)
