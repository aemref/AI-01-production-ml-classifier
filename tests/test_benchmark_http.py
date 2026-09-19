from io import BytesIO
import json
import subprocess
import sys
from threading import Lock
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

import src.benchmark_http as benchmark_http
from src.benchmark_http import (
    HttpBenchmarkResult,
    RequestObservation,
    normalize_base_url,
    perform_prediction_request,
    run_http_benchmark,
    summarize_observations,
)


class StubResponse:
    def __init__(self, status=200, body=None):
        self.status = status
        self.body = body or (
            b'{"label":0,"label_name":"malignant","confidence":0.9,'
            b'"malignant_probability":0.9,"benign_probability":0.1,'
            b'"model_version":"test-model"}'
        )
        self.was_read = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        self.was_read = True
        return self.body


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
    "body",
    [
        b"<html>not the classifier</html>",
        b"{}",
        b'{"label":0,"label_name":"malignant","confidence":0.9,'
        b'"malignant_probability":0.9,"benign_probability":0.9,'
        b'"model_version":"test-model"}',
    ],
)
def test_prediction_request_rejects_invalid_success_body(body):
    timestamps = iter([0.0, 0.1])
    result = perform_prediction_request(
        "http://localhost:8000",
        timeout_seconds=2.0,
        clock=lambda: next(timestamps),
        opener=lambda *_args, **_kwargs: StubResponse(body=body),
    )

    assert result == RequestObservation(0.1, 200, "invalid_response")


def test_summary_counts_invalid_success_body_as_failure():
    result = summarize_observations(
        [RequestObservation(0.01, 200, "invalid_response")],
        wall_seconds=0.02,
    )

    assert result.success_count == 0
    assert result.failure_count == 1
    assert result.failure_types == {"invalid_response": 1}


def test_warmup_rejects_invalid_success_body():
    with pytest.raises(RuntimeError, match="Warmup request failed: invalid_response"):
        run_http_benchmark(
            "http://localhost:8000",
            request_count=1,
            warmup_count=1,
            requester=lambda *_args, **_kwargs: RequestObservation(
                0.01, 200, "invalid_response"
            ),
        )


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


def test_http_benchmark_excludes_warmup_and_counts_concurrent_failures():
    responses = iter(
        [
            RequestObservation(0.001, 200),
            RequestObservation(0.010, 200),
            RequestObservation(0.020, 503),
            RequestObservation(0.030, None, "timeout"),
        ]
    )
    timestamps = iter([10.0, 10.05])

    def requester(base_url, *, timeout_seconds):
        assert base_url == "http://localhost:8000"
        assert timeout_seconds == 1.0
        return next(responses)

    result = run_http_benchmark(
        "http://localhost:8000/",
        request_count=3,
        warmup_count=1,
        concurrency=2,
        timeout_seconds=1.0,
        requester=requester,
        clock=lambda: next(timestamps),
    )

    assert result.request_count == 3
    assert result.failure_count == 2
    assert result.throughput_requests_per_second == pytest.approx(60.0)
    assert result.failure_types == {"http_503": 1, "timeout": 1}


def test_http_benchmark_never_exceeds_requested_concurrency():
    lock = Lock()
    active = 0
    maximum_active = 0

    def requester(_base_url, *, timeout_seconds):
        nonlocal active, maximum_active
        assert timeout_seconds == 2.0
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        sleep(0.01)
        with lock:
            active -= 1
        return RequestObservation(0.01, 200)

    timestamps = iter([0.0, 0.08])
    result = run_http_benchmark(
        "http://localhost:8000",
        request_count=8,
        warmup_count=0,
        concurrency=3,
        requester=requester,
        clock=lambda: next(timestamps),
    )

    assert result.success_count == 8
    assert maximum_active == 3


def test_http_benchmark_fails_fast_when_warmup_cannot_succeed():
    def requester(_base_url, *, timeout_seconds):
        return RequestObservation(timeout_seconds, None, "connection_error")

    with pytest.raises(RuntimeError, match="Warmup request failed: connection_error"):
        run_http_benchmark(
            "http://localhost:8000",
            request_count=1,
            warmup_count=1,
            requester=requester,
        )


@pytest.mark.parametrize(
    ("request_count", "warmup_count", "concurrency"),
    [(0, 0, 1), (1, -1, 1), (1, 0, 0), (1, 0, 65)],
)
def test_http_benchmark_rejects_unsafe_bounds(
    request_count,
    warmup_count,
    concurrency,
):
    with pytest.raises(ValueError):
        run_http_benchmark(
            "http://localhost:8000",
            request_count=request_count,
            warmup_count=warmup_count,
            concurrency=concurrency,
        )


def test_cli_emits_configuration_with_machine_readable_result(monkeypatch, capsys):
    expected = HttpBenchmarkResult(
        request_count=10,
        success_count=10,
        failure_count=0,
        total_wall_seconds=0.1,
        throughput_requests_per_second=100.0,
        latency_ms_min=1.0,
        latency_ms_median=2.0,
        latency_ms_p95=3.0,
        latency_ms_max=4.0,
        status_counts={"200": 10},
        failure_types={},
    )
    captured = {}

    def fake_run(target, **options):
        captured["target"] = target
        captured["options"] = options
        return expected

    monkeypatch.setattr(benchmark_http, "run_http_benchmark", fake_run)
    benchmark_http.main(
        [
            "--base-url",
            "http://localhost:9000/",
            "--requests",
            "10",
            "--warmup",
            "2",
            "--concurrency",
            "5",
            "--timeout",
            "1.5",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert captured == {
        "target": "http://localhost:9000",
        "options": {
            "request_count": 10,
            "warmup_count": 2,
            "concurrency": 5,
            "timeout_seconds": 1.5,
        },
    }
    assert report["target"] == "http://localhost:9000"
    assert report["warmup_count"] == 2
    assert report["concurrency"] == 5
    assert report["failure_count"] == 0


def test_cli_returns_usage_error_when_target_is_unavailable(monkeypatch, capsys):
    def fake_run(*_args, **_kwargs):
        raise RuntimeError("Warmup request failed: connection_error")

    monkeypatch.setattr(benchmark_http, "run_http_benchmark", fake_run)
    with pytest.raises(SystemExit) as exit_info:
        benchmark_http.main([])

    assert exit_info.value.code == 2
    assert "Warmup request failed: connection_error" in capsys.readouterr().err


def test_cli_reports_measured_failures_and_exits_unsuccessfully(monkeypatch, capsys):
    result = HttpBenchmarkResult(
        request_count=2,
        success_count=1,
        failure_count=1,
        total_wall_seconds=0.2,
        throughput_requests_per_second=10.0,
        latency_ms_min=2.0,
        latency_ms_median=3.0,
        latency_ms_p95=4.0,
        latency_ms_max=4.0,
        status_counts={"200": 2},
        failure_types={"invalid_response": 1},
    )
    monkeypatch.setattr(benchmark_http, "run_http_benchmark", lambda *_args, **_kwargs: result)

    with pytest.raises(SystemExit) as exit_info:
        benchmark_http.main([])

    assert exit_info.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["request_count"] == 2
    assert report["failure_count"] == 1
    assert report["failure_types"] == {"invalid_response": 1}


def test_module_entrypoint_summarizes_failed_request():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.benchmark_http",
            "--base-url",
            "http://127.0.0.1:1",
            "--requests",
            "1",
            "--warmup",
            "0",
            "--timeout",
            "0.2",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert completed.returncode == 1
    assert json.loads(completed.stdout)["failure_count"] == 1
