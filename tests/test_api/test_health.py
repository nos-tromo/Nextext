"""Tests for ``GET /api/v1/health``."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_health_returns_ok_with_unreachable_inference(
    api_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Health should report status=ok even when inference is unreachable."""
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response = api_client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["inference"] is False
    assert isinstance(body["version"], str) and body["version"]


def test_health_marks_inference_reachable_when_get_health_returns_true(
    api_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``InferencePipeline.get_health`` is True, the field should be True."""
    from nextext.api.routes import health as health_module

    class _StubPipeline:
        def get_health(self) -> bool:
            return True

    monkeypatch.setattr(health_module, "InferencePipeline", _StubPipeline)

    response = api_client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["inference"] is True


def test_health_swallows_inference_exception(
    api_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The endpoint must never 500 because the inference check raised."""
    from nextext.api.routes import health as health_module

    class _BrokenPipeline:
        def get_health(self) -> bool:
            raise RuntimeError("simulated provider failure")

    monkeypatch.setattr(health_module, "InferencePipeline", _BrokenPipeline)

    response = api_client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["inference"] is False
