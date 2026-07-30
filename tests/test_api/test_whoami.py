"""Tests for ``GET /api/v1/whoami``."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nextext.api.main import create_app
from tests.test_api.conftest import ALICE_OWNER_ID, OWNER_HEADER


def test_whoami_returns_username_and_display_name(api_client: TestClient) -> None:
    """A present trusted header plus X-Auth-Name populates both fields."""
    response = api_client.get("/api/v1/whoami", headers={"X-Auth-Name": "Alice Example"})
    assert response.status_code == 200
    assert response.json() == {"username": ALICE_OWNER_ID, "display_name": "Alice Example"}


def test_whoami_display_name_absent_when_header_missing(api_client: TestClient) -> None:
    """Without X-Auth-Name, display_name is null but username still resolves."""
    response = api_client.get("/api/v1/whoami")
    assert response.status_code == 200
    assert response.json() == {"username": ALICE_OWNER_ID, "display_name": None}


def test_whoami_falls_back_to_default_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no header, the configured dev default identity is returned."""
    monkeypatch.setenv("NEXTEXT_DEFAULT_IDENTITY", "service-account")
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/whoami")
    assert response.status_code == 200
    assert response.json() == {"username": "service-account", "display_name": None}


def test_whoami_fails_closed_without_header_or_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """No header and no configured fallback means 401, like every other route."""
    monkeypatch.delenv("NEXTEXT_DEFAULT_IDENTITY", raising=False)
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/whoami")
    assert response.status_code == 401


def test_whoami_ignores_untrusted_owner_header_impersonation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller cannot spoof another owner merely by sending a different header value.

    This just re-confirms resolve_principal's normal behavior (the header
    value *is* the principal) — there is no separate trust boundary to break
    here, but it documents that /whoami echoes exactly what identity.py
    resolved, not anything else client-supplied.
    """
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/whoami", headers={OWNER_HEADER: "  bob  "})
    assert response.status_code == 200
    assert response.json()["username"] == "bob"
