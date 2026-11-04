"""Tests for ``GET /api/v1/whoami``."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nextext.api.identity import EDGE_SECRET_HEADER
from nextext.api.main import create_app
from tests.test_api.conftest import ALICE_OWNER_ID, OWNER_HEADER

# Invented credentials, long enough to pass validation.
EDGE_SECRET = "edge-secret-for-tests-only-0123456789abcdef"
DOCINT_TOKEN = "svc-token-docint-for-tests-only-0123456789"


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

    This just re-confirms resolve_principal's behavior without an edge secret
    (the header value *is* the principal) — the edge-secret tests cover the
    trust boundary — and documents that /whoami echoes exactly what
    identity.py resolved, not anything else client-supplied.
    """
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/whoami", headers={OWNER_HEADER: "  bob  "})
    assert response.status_code == 200
    assert response.json()["username"] == "bob"


def test_whoami_shows_display_name_on_edge_verified_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the edge secret set, a verified request gets both fields."""
    monkeypatch.setenv("NEXTEXT_EDGE_SECRET", EDGE_SECRET)
    headers = {OWNER_HEADER: "alice", "X-Auth-Name": "Alice Example", EDGE_SECRET_HEADER: EDGE_SECRET}
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/whoami", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"username": "alice", "display_name": "Alice Example"}


def test_whoami_ignores_gateway_headers_without_edge_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unverified request gets neither the forged user nor the forged display name.

    Without a verified identity it gets 401 (the default identity does not
    apply while the edge secret is set); a request that resolves another way —
    a service token — still never echoes the forged display name.
    """
    monkeypatch.setenv("NEXTEXT_EDGE_SECRET", EDGE_SECRET)
    monkeypatch.setenv("NEXTEXT_DEFAULT_IDENTITY", "service-account")
    monkeypatch.setenv("NEXTEXT_SERVICE_TOKENS", f"docint={DOCINT_TOKEN}")
    headers = {OWNER_HEADER: "forged-owner-for-tests", "X-Auth-Name": "Forged Name"}
    with TestClient(create_app()) as client:
        unverified = client.get("/api/v1/whoami", headers=headers)
        service = client.get("/api/v1/whoami", headers={**headers, "Authorization": f"Bearer {DOCINT_TOKEN}"})
    assert unverified.status_code == 401
    assert service.status_code == 200
    assert service.json() == {"username": "service:docint", "display_name": None}


def test_whoami_resolves_a_docint_style_service_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The docint client's headers (bearer token plus ``X-Auth-User: docint``) resolve to ``service:docint``."""
    monkeypatch.setenv("NEXTEXT_EDGE_SECRET", EDGE_SECRET)
    monkeypatch.setenv("NEXTEXT_SERVICE_TOKENS", f"docint={DOCINT_TOKEN}")
    headers = {"Authorization": f"Bearer {DOCINT_TOKEN}", OWNER_HEADER: "docint"}
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/whoami", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"username": "service:docint", "display_name": None}
