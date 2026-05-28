"""Tests for the ``X-Owner-Id`` header dependency."""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from nextext.api.identity import OWNER_HEADER, get_owner_id


def _build_app() -> FastAPI:
    """Return a tiny FastAPI app that echoes the resolved owner id.

    Returns:
        FastAPI: An app with one ``GET /whoami`` route protected by the
            :func:`get_owner_id` dependency.
    """
    app = FastAPI()

    @app.get("/whoami")
    def whoami(owner_id: str = Depends(get_owner_id)) -> dict[str, str]:
        return {"owner": owner_id}

    return app


def test_valid_header_returns_owner() -> None:
    """A well-formed header value should reach the handler unchanged."""
    app = _build_app()
    valid_id = "a" * 32
    with TestClient(app) as client:
        response = client.get("/whoami", headers={OWNER_HEADER: valid_id})
        assert response.status_code == 200
        assert response.json() == {"owner": valid_id}


def test_missing_header_returns_400() -> None:
    """Requests without the header must be rejected with 400."""
    app = _build_app()
    with TestClient(app) as client:
        response = client.get("/whoami")
        assert response.status_code == 400
        assert "X-Owner-Id" in response.json()["detail"]


@pytest.mark.parametrize(
    "value",
    [
        "",
        "not-a-uuid",
        "g" * 32,  # 32 chars but not hex.
        "a" * 31,  # too short.
        "a" * 33,  # too long.
        "A" * 32,  # uppercase hex is also rejected — keeps DB rows consistent.
    ],
)
def test_invalid_header_returns_400(value: str) -> None:
    """Malformed header values must be rejected with 400."""
    app = _build_app()
    with TestClient(app) as client:
        response = client.get("/whoami", headers={OWNER_HEADER: value})
        assert response.status_code == 400


def test_two_browsers_resolve_to_distinct_owners() -> None:
    """Two clients sending different headers see different owners."""
    app = _build_app()
    alice = "a" * 32
    bob = "b" * 32
    with TestClient(app) as client:
        first = client.get("/whoami", headers={OWNER_HEADER: alice}).json()["owner"]
        second = client.get("/whoami", headers={OWNER_HEADER: bob}).json()["owner"]
    assert first == alice
    assert second == bob
    assert first != second
