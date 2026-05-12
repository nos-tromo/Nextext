"""Tests for :class:`nextext.api.identity.IdentityMiddleware`."""

from __future__ import annotations

import uuid

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from nextext.api.identity import COOKIE_NAME, IdentityMiddleware, get_owner_id


def _build_app() -> FastAPI:
    """Build a tiny FastAPI app that echoes the resolved owner id.

    Returns:
        FastAPI: An app with :class:`IdentityMiddleware` installed and a
            single ``GET /whoami`` endpoint.
    """
    app = FastAPI()
    app.add_middleware(IdentityMiddleware)

    @app.get("/whoami")
    def whoami(request: Request) -> dict[str, str]:
        return {"owner": get_owner_id(request)}

    return app


def test_first_request_mints_cookie_and_returns_owner() -> None:
    """The first request from a clean browser should mint a session cookie."""
    app = _build_app()
    with TestClient(app) as client:
        response = client.get("/whoami")
        assert response.status_code == 200
        owner = response.json()["owner"]
        assert uuid.UUID(hex=owner)
        assert COOKIE_NAME in response.cookies
        assert response.cookies[COOKIE_NAME] == owner


def test_repeated_requests_reuse_same_cookie_value() -> None:
    """Subsequent requests on the same TestClient must not rotate the id."""
    app = _build_app()
    with TestClient(app) as client:
        first = client.get("/whoami").json()["owner"]
        second = client.get("/whoami").json()["owner"]
        assert first == second


def test_two_clients_get_distinct_owner_ids() -> None:
    """Separate clients (browsers) must receive distinct owner ids."""
    app = _build_app()
    with TestClient(app) as alice, TestClient(app) as bob:
        alice_id = alice.get("/whoami").json()["owner"]
        bob_id = bob.get("/whoami").json()["owner"]
    assert alice_id != bob_id


def test_malformed_cookie_is_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spoofed or malformed cookies must not poison the owner_id."""
    app = _build_app()
    with TestClient(app) as client:
        response = client.get(
            "/whoami",
            cookies={COOKIE_NAME: "not-a-uuid"},
        )
        owner = response.json()["owner"]
        assert uuid.UUID(hex=owner)
        # The server replaced the cookie on its way out.
        assert COOKIE_NAME in response.cookies
        assert response.cookies[COOKIE_NAME] != "not-a-uuid"


def test_secure_flag_can_be_enabled_via_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``NEXTEXT_SESSION_COOKIE_SECURE`` should propagate to the Set-Cookie header."""
    monkeypatch.setenv("NEXTEXT_SESSION_COOKIE_SECURE", "1")
    app = FastAPI()
    app.add_middleware(IdentityMiddleware)

    @app.get("/whoami")
    def whoami(request: Request) -> dict[str, str]:
        return {"owner": get_owner_id(request)}

    with TestClient(app) as client:
        response = client.get("/whoami")
        set_cookie = response.headers.get("set-cookie", "")
        lowered = set_cookie.lower()
        assert "secure" in lowered
        assert "httponly" in lowered
        assert "samesite=lax" in lowered
