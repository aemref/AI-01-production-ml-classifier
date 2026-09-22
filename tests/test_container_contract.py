from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_container_runs_api_as_non_root_with_healthcheck():
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert dockerfile.startswith("FROM python:3.11-slim")
    assert "USER app" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "COPY artifacts ./artifacts" in dockerfile
    assert '"src.api:app"' in dockerfile
    assert '"--host", "0.0.0.0"' in dockerfile


def test_container_context_excludes_local_and_development_artifacts():
    excluded = set((ROOT / ".dockerignore").read_text().splitlines())

    assert {".git", ".venv", ".pytest_cache", "tests", "notebooks"} <= excluded


def test_ci_benchmarks_the_running_container_over_external_http():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    start = workflow.index("docker run --rm -d")
    benchmark = workflow.index("python -m src.benchmark_http")
    stop = workflow.index("docker stop ai01-classifier-ci")

    assert start < benchmark < stop
    assert "--base-url http://127.0.0.1:8000" in workflow
    assert "--concurrency 4" in workflow
    assert "--timeout 2" in workflow


def test_ci_verifies_artifact_demo_and_release_evidence_before_packaging():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    artifact = workflow.index("python -m src.verify_artifact")
    demo = workflow.index("python -m src.demo --compact")
    release = workflow.index("python -m src.release_readiness")
    container = workflow.index("docker build --tag ai01-classifier:ci")

    assert artifact < demo < release < container
