"""Tests for the Prometheus ``GET /metrics`` endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_metrics_returns_prometheus_exposition(api_client: TestClient) -> None:
    """The metrics endpoint should expose Prometheus text-format counters."""
    response = api_client.get("/metrics")

    assert response.status_code == 200
    assert "http_requests_total" in response.text
