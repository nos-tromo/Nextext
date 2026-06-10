"""Tests for the request-principal resolver (trusted header + env fallback)."""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from nextext.api.identity import resolve_principal
from nextext.utils.env_cfg import load_principal_env

DEFAULT_HEADER = "X-Auth-User"


def _build_app() -> FastAPI:
    """Return a tiny FastAPI app that echoes the resolved principal.

    Returns:
        FastAPI: An app with one ``GET /whoami`` route protected by the
            :func:`resolve_principal` dependency.
    """
    app = FastAPI()

    @app.get("/whoami")
    def whoami(owner_id: str = Depends(resolve_principal)) -> dict[str, str]:
        return {"owner": owner_id}

    return app


@pytest.fixture(autouse=True)
def _clean_principal_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each test with the identity env vars unset.

    Args:
        monkeypatch: Pytest environment patcher.
    """
    monkeypatch.delenv("NEXTEXT_AUTH_HEADER", raising=False)
    monkeypatch.delenv("NEXTEXT_DEFAULT_IDENTITY", raising=False)


def test_trusted_header_returns_its_value() -> None:
    """The trusted header value reaches the handler verbatim (stripped)."""
    app = _build_app()
    with TestClient(app) as client:
        response = client.get("/whoami", headers={DEFAULT_HEADER: "  alice  "})
    assert response.status_code == 200
    assert response.json() == {"owner": "alice"}


def test_free_form_identity_is_accepted() -> None:
    """Any non-empty identity is accepted (no UUID format requirement)."""
    app = _build_app()
    with TestClient(app) as client:
        response = client.get("/whoami", headers={DEFAULT_HEADER: "dev-user-42"})
    assert response.status_code == 200
    assert response.json() == {"owner": "dev-user-42"}


def test_custom_header_name_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    """``NEXTEXT_AUTH_HEADER`` overrides which header is trusted."""
    monkeypatch.setenv("NEXTEXT_AUTH_HEADER", "X-Dev-User")
    app = _build_app()
    with TestClient(app) as client:
        ignored = client.get("/whoami", headers={DEFAULT_HEADER: "alice"})
        honoured = client.get("/whoami", headers={"X-Dev-User": "bob"})
    assert ignored.status_code == 401
    assert honoured.status_code == 200
    assert honoured.json() == {"owner": "bob"}


def test_missing_header_falls_back_to_default_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no header, the configured default identity is returned."""
    monkeypatch.setenv("NEXTEXT_DEFAULT_IDENTITY", "service-account")
    app = _build_app()
    with TestClient(app) as client:
        response = client.get("/whoami")
    assert response.status_code == 200
    assert response.json() == {"owner": "service-account"}


def test_present_header_wins_over_default_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """A present header takes precedence over the default identity."""
    monkeypatch.setenv("NEXTEXT_DEFAULT_IDENTITY", "service-account")
    app = _build_app()
    with TestClient(app) as client:
        response = client.get("/whoami", headers={DEFAULT_HEADER: "alice"})
    assert response.status_code == 200
    assert response.json() == {"owner": "alice"}


def test_no_header_and_no_default_is_401() -> None:
    """With neither header nor default identity, the resolver fails closed."""
    app = _build_app()
    with TestClient(app) as client:
        response = client.get("/whoami")
    assert response.status_code == 401


def test_blank_header_is_treated_as_absent() -> None:
    """A whitespace-only header is ignored and falls through to 401."""
    app = _build_app()
    with TestClient(app) as client:
        response = client.get("/whoami", headers={DEFAULT_HEADER: "   "})
    assert response.status_code == 401


def test_two_browsers_resolve_to_distinct_owners() -> None:
    """Two clients sending different headers see different owners."""
    app = _build_app()
    with TestClient(app) as client:
        first = client.get("/whoami", headers={DEFAULT_HEADER: "a" * 32}).json()["owner"]
        second = client.get("/whoami", headers={DEFAULT_HEADER: "b" * 32}).json()["owner"]
    assert first != second


def test_blank_default_identity_normalises_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A whitespace-only ``NEXTEXT_DEFAULT_IDENTITY`` is treated as unset."""
    monkeypatch.setenv("NEXTEXT_DEFAULT_IDENTITY", "   ")
    assert load_principal_env().default_identity is None
