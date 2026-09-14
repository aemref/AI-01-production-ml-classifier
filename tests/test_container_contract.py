from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_container_runs_api_as_non_root_with_healthcheck():
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert dockerfile.startswith("FROM python:3.11-slim")
    assert "USER app" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert '"src.api:app"' in dockerfile
    assert '"--host", "0.0.0.0"' in dockerfile


def test_container_context_excludes_local_and_development_artifacts():
    excluded = set((ROOT / ".dockerignore").read_text().splitlines())

    assert {".git", ".venv", ".pytest_cache", "tests", "notebooks"} <= excluded
