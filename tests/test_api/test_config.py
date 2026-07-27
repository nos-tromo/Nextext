"""Tests for ``GET /api/v1/config``."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_get_config_language(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /api/v1/config reports the resolved UI language."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "de")

    response = api_client.get("/api/v1/config")

    assert response.status_code == 200
    assert response.json() == {"language": "de"}
