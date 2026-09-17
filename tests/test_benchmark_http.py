from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from src.benchmark_http import (
    RequestObservation,
    normalize_base_url,
    perform_prediction_request,
    summarize_observations,
)


class StubResponse:
    def __init__(self, status=200):
        self.status = status
        self.was_read = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        self.was_read = True
        return b"{}"


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


def test_prediction_request_sends_json_and_records_http_status():
    response = StubResponse()
    captured = {}
    timestamps = iter([1.0, 1.025])

    def opener(request, *, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return response

    result = perform_prediction_request(
        "http://127.0.0.1:8000/",
        timeout_seconds=1.5,
        clock=lambda: next(timestamps),
        opener=opener,
    )

    request: Request = captured["request"]
    assert request.full_url == "http://127.0.0.1:8000/predict"
    assert request.method == "POST"
    assert request.headers["Content-type"] == "application/json"
    assert request.data == b'{"feature_a": 17.99, "feature_b": 10.38}'
    assert captured["timeout"] == 1.5
    assert response.was_read is True
    assert result == RequestObservation(0.02499999999999991, 200)


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (
            HTTPError(
                "http://localhost/predict",
                503,
                "unavailable",
                {},
                BytesIO(),
            ),
            RequestObservation(0.1, 503),
        ),
        (URLError(ConnectionRefusedError()), RequestObservation(0.1, None, "connection_error")),
        (TimeoutError(), RequestObservation(0.1, None, "timeout")),
    ],
)
def test_prediction_request_classifies_failures(raised, expected):
    timestamps = iter([0.0, 0.1])

    def opener(_request, *, timeout):
        assert timeout == 2.0
        raise raised

    result = perform_prediction_request(
        "http://localhost:8000",
        timeout_seconds=2.0,
        clock=lambda: next(timestamps),
        opener=opener,
    )

    assert result == expected


@pytest.mark.parametrize(
    "url",
    [
        "localhost:8000",
        "ftp://localhost/model",
        "http://user:secret@localhost",
        "http://localhost?token=secret",
        "http://localhost#fragment",
    ],
)
def test_base_url_rejects_ambiguous_or_sensitive_targets(url):
    with pytest.raises(ValueError, match="Base URL"):
        normalize_base_url(url)
