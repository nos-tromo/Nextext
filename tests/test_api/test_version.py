"""Tests for ``GET /api/v1/version``."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_version_returns_package_version(api_client: TestClient) -> None:
    """GET /api/v1/version returns a non-empty version string."""
    resp = api_client.get("/api/v1/version")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["version"], str) and body["version"]
