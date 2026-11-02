"""Tests for the request-principal resolver and its startup check.

Covers service tokens, the trusted header with and without the edge secret,
the env fallback, and what reaches the logs.
"""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from nextext.api import main as main_module
from nextext.api.identity import EDGE_SECRET_HEADER, resolve_principal
from nextext.api.main import create_app
from nextext.utils.env_cfg import PrincipalConfigError, load_principal_env

DEFAULT_HEADER = "X-Auth-User"

# Invented credentials and owners, long enough to pass validation.
EDGE_SECRET = "edge-secret-for-tests-only-0123456789abcdef"
WRONG_EDGE_SECRET = "wrong-edge-secret-for-tests-only-0123456789"
DOCINT_TOKEN = "svc-token-docint-for-tests-only-0123456789"
OTHER_TOKEN = "svc-token-other-for-tests-only-0123456789ab"
UNKNOWN_TOKEN = "svc-token-unknown-for-tests-only-0123456789"
FORGED_OWNER = "forged-owner-for-tests"
STARTUP_WARNING = "NEXTEXT_EDGE_SECRET is unset"


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


def _set_service_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure two service tokens, ``docint`` and ``other``.

    Args:
        monkeypatch: Pytest environment patcher.
    """
    monkeypatch.setenv("NEXTEXT_SERVICE_TOKENS", f"docint={DOCINT_TOKEN},other={OTHER_TOKEN}")


# --- Edge secret --------------------------------------------------------------


def test_edge_verified_request_honours_the_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the edge secret set, a matching ``X-Edge-Secret`` lets the header through."""
    monkeypatch.setenv("NEXTEXT_EDGE_SECRET", EDGE_SECRET)
    with TestClient(_build_app()) as client:
        response = client.get("/whoami", headers={DEFAULT_HEADER: "alice", EDGE_SECRET_HEADER: EDGE_SECRET})
    assert response.status_code == 200
    assert response.json() == {"owner": "alice"}


@pytest.mark.parametrize(
    "edge_headers",
    [
        pytest.param({}, id="missing"),
        pytest.param({EDGE_SECRET_HEADER: WRONG_EDGE_SECRET}, id="wrong"),
        pytest.param({EDGE_SECRET_HEADER: ""}, id="empty"),
        pytest.param({EDGE_SECRET_HEADER: EDGE_SECRET[:-1]}, id="prefix"),
    ],
)
def test_unverified_header_is_ignored(monkeypatch: pytest.MonkeyPatch, edge_headers: dict[str, str]) -> None:
    """Without a matching ``X-Edge-Secret`` the header counts as absent: 401.

    Args:
        monkeypatch: Pytest environment patcher.
        edge_headers: The ``X-Edge-Secret`` header to send, if any.
    """
    monkeypatch.setenv("NEXTEXT_EDGE_SECRET", EDGE_SECRET)
    with TestClient(_build_app()) as client:
        response = client.get("/whoami", headers={DEFAULT_HEADER: FORGED_OWNER, **edge_headers})
    assert response.status_code == 401


def test_non_ascii_edge_secret_is_a_mismatch_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-ASCII ``X-Edge-Secret`` yields 401, never a server error."""
    monkeypatch.setenv("NEXTEXT_EDGE_SECRET", EDGE_SECRET)
    raw_headers = [
        (DEFAULT_HEADER.encode(), FORGED_OWNER.encode()),
        (EDGE_SECRET_HEADER.encode(), b"\xe9" * len(EDGE_SECRET)),
    ]
    with TestClient(_build_app()) as client:
        response = client.get("/whoami", headers=raw_headers)
    assert response.status_code == 401


def test_unverified_header_falls_back_to_default_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ignored header falls through to the header-less behaviour."""
    monkeypatch.setenv("NEXTEXT_EDGE_SECRET", EDGE_SECRET)
    monkeypatch.setenv("NEXTEXT_DEFAULT_IDENTITY", "service-account")
    with TestClient(_build_app()) as client:
        response = client.get("/whoami", headers={DEFAULT_HEADER: FORGED_OWNER})
    assert response.status_code == 200
    assert response.json() == {"owner": "service-account"}


def test_ignored_header_is_logged_without_values(monkeypatch: pytest.MonkeyPatch, log_lines: list[str]) -> None:
    """The warning names method and path only — no header value, secret or query.

    Args:
        monkeypatch: Pytest environment patcher.
        log_lines: Captured loguru messages.
    """
    monkeypatch.setenv("NEXTEXT_EDGE_SECRET", EDGE_SECRET)
    with TestClient(_build_app()) as client:
        response = client.get(
            "/whoami?token=query-value-for-tests",
            headers={DEFAULT_HEADER: FORGED_OWNER, EDGE_SECRET_HEADER: WRONG_EDGE_SECRET},
        )
    assert response.status_code == 401
    assert len(log_lines) == 1
    assert "GET /whoami:" in log_lines[0]
    assert DEFAULT_HEADER in log_lines[0]
    for value in (FORGED_OWNER, WRONG_EDGE_SECRET, EDGE_SECRET, "query-value-for-tests", "token="):
        assert value not in log_lines[0]


def test_verified_and_headerless_requests_log_nothing(monkeypatch: pytest.MonkeyPatch, log_lines: list[str]) -> None:
    """Only an identity header that was actually ignored is reported.

    Args:
        monkeypatch: Pytest environment patcher.
        log_lines: Captured loguru messages.
    """
    monkeypatch.setenv("NEXTEXT_EDGE_SECRET", EDGE_SECRET)
    with TestClient(_build_app()) as client:
        verified = client.get("/whoami", headers={DEFAULT_HEADER: "alice", EDGE_SECRET_HEADER: EDGE_SECRET})
        headerless = client.get("/whoami", headers={EDGE_SECRET_HEADER: WRONG_EDGE_SECRET})
    assert verified.status_code == 200
    assert headerless.status_code == 401
    assert log_lines == []


def test_logged_path_is_percent_encoded(monkeypatch: pytest.MonkeyPatch, log_lines: list[str]) -> None:
    """A control character in the path cannot forge a second log line.

    Args:
        monkeypatch: Pytest environment patcher.
        log_lines: Captured loguru messages.
    """
    monkeypatch.setenv("NEXTEXT_EDGE_SECRET", EDGE_SECRET)
    app = FastAPI()

    @app.get("/items/{item_id}")
    def read_item(item_id: str, owner_id: str = Depends(resolve_principal)) -> dict[str, str]:
        return {"owner": owner_id, "item": item_id}

    with TestClient(app) as client:
        response = client.get("/items/a%0Afake-log-line", headers={DEFAULT_HEADER: FORGED_OWNER})
    assert response.status_code == 401
    assert len(log_lines) == 1
    assert "/items/a%0Afake-log-line" in log_lines[0]
    assert "\n" not in log_lines[0].rstrip()


# --- Service tokens -----------------------------------------------------------


@pytest.mark.parametrize(("token", "name"), [(DOCINT_TOKEN, "docint"), (OTHER_TOKEN, "other")])
def test_service_token_resolves_to_its_name_not_the_header(
    monkeypatch: pytest.MonkeyPatch, token: str, name: str
) -> None:
    """A matching bearer token wins over the header: a service cannot act as a user.

    Args:
        monkeypatch: Pytest environment patcher.
        token: The bearer token to present.
        name: The service name it is configured under.
    """
    _set_service_tokens(monkeypatch)
    with TestClient(_build_app()) as client:
        response = client.get("/whoami", headers={"Authorization": f"Bearer {token}", DEFAULT_HEADER: "alice"})
    assert response.status_code == 200
    assert response.json() == {"owner": name}


def test_service_token_needs_no_edge_secret_and_beats_a_verified_header(
    monkeypatch: pytest.MonkeyPatch, log_lines: list[str]
) -> None:
    """Service calls skip the edge check, and the header they carry is ignored silently.

    Args:
        monkeypatch: Pytest environment patcher.
        log_lines: Captured loguru messages.
    """
    monkeypatch.setenv("NEXTEXT_EDGE_SECRET", EDGE_SECRET)
    _set_service_tokens(monkeypatch)
    bearer = {"Authorization": f"Bearer {DOCINT_TOKEN}"}
    with TestClient(_build_app()) as client:
        direct = client.get("/whoami", headers={**bearer, DEFAULT_HEADER: "docint"})
        verified = client.get("/whoami", headers={**bearer, DEFAULT_HEADER: "alice", EDGE_SECRET_HEADER: EDGE_SECRET})
    assert direct.json() == {"owner": "docint"}
    assert verified.json() == {"owner": "docint"}
    assert log_lines == []


def test_bearer_scheme_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    """The auth scheme matches regardless of case."""
    _set_service_tokens(monkeypatch)
    with TestClient(_build_app()) as client:
        response = client.get("/whoami", headers={"Authorization": f"bearer {DOCINT_TOKEN}"})
    assert response.json() == {"owner": "docint"}


@pytest.mark.parametrize(
    "authorization",
    [
        pytest.param(f"Bearer {UNKNOWN_TOKEN}", id="unknown-token"),
        pytest.param(f"Basic {DOCINT_TOKEN}", id="other-scheme"),
        pytest.param(DOCINT_TOKEN, id="no-scheme"),
        pytest.param("Bearer", id="no-credential"),
    ],
)
def test_non_matching_authorization_falls_through(monkeypatch: pytest.MonkeyPatch, authorization: str) -> None:
    """Anything but a matching bearer token continues with the header path.

    Args:
        monkeypatch: Pytest environment patcher.
        authorization: The ``Authorization`` header value to send.
    """
    _set_service_tokens(monkeypatch)
    headers = {"Authorization": authorization, DEFAULT_HEADER: "alice"}
    with TestClient(_build_app()) as client:
        trusted = client.get("/whoami", headers=headers)
        monkeypatch.setenv("NEXTEXT_EDGE_SECRET", EDGE_SECRET)
        enforced = client.get("/whoami", headers=headers)
    assert trusted.json() == {"owner": "alice"}
    assert enforced.status_code == 401


def test_credentials_never_reach_the_logs(monkeypatch: pytest.MonkeyPatch, log_lines: list[str]) -> None:
    """Neither the secret, a token, nor a forged identity is ever logged.

    Args:
        monkeypatch: Pytest environment patcher.
        log_lines: Captured loguru messages.
    """
    monkeypatch.setenv("NEXTEXT_EDGE_SECRET", EDGE_SECRET)
    _set_service_tokens(monkeypatch)
    with TestClient(_build_app()) as client:
        client.get("/whoami", headers={"Authorization": f"Bearer {DOCINT_TOKEN}", DEFAULT_HEADER: FORGED_OWNER})
        client.get("/whoami", headers={"Authorization": f"Bearer {UNKNOWN_TOKEN}", DEFAULT_HEADER: FORGED_OWNER})
        client.get("/whoami", headers={EDGE_SECRET_HEADER: WRONG_EDGE_SECRET, DEFAULT_HEADER: FORGED_OWNER})
    assert len(log_lines) == 2
    logged = "".join(log_lines)
    for value in (EDGE_SECRET, WRONG_EDGE_SECRET, DOCINT_TOKEN, OTHER_TOKEN, UNKNOWN_TOKEN, FORGED_OWNER):
        assert value not in logged


# --- Startup --------------------------------------------------------------------


def test_startup_warns_once_without_edge_secret_in_production_shape(
    monkeypatch: pytest.MonkeyPatch, log_lines: list[str]
) -> None:
    """No secret and no default identity: one warning at startup, none per request.

    Args:
        monkeypatch: Pytest patcher; keeps the app's logging setup from
            removing the capture sink.
        log_lines: Captured loguru messages.
    """
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    with TestClient(create_app()) as client:
        client.get("/api/v1/whoami", headers={DEFAULT_HEADER: "alice"})
        client.get("/api/v1/whoami", headers={DEFAULT_HEADER: "alice"})
    warnings = [line for line in log_lines if STARTUP_WARNING in line]
    assert len(warnings) == 1
    assert "edge-plane docs/identity-contract.md" in warnings[0]


@pytest.mark.parametrize(
    ("name", "value"),
    [("NEXTEXT_EDGE_SECRET", EDGE_SECRET), ("NEXTEXT_DEFAULT_IDENTITY", "service-account")],
)
def test_startup_does_not_warn_with_edge_secret_or_dev_identity(
    monkeypatch: pytest.MonkeyPatch, log_lines: list[str], name: str, value: str
) -> None:
    """An edge secret, or a dev default identity, silences the startup warning.

    Args:
        monkeypatch: Pytest patcher.
        log_lines: Captured loguru messages.
        name: The variable to set.
        value: Its value.
    """
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    monkeypatch.setenv(name, value)
    with TestClient(create_app()):
        pass
    assert not any(STARTUP_WARNING in line for line in log_lines)
    assert EDGE_SECRET not in "".join(log_lines)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        pytest.param("NEXTEXT_EDGE_SECRET", "short-edge-secret-for-tests", id="short-secret"),
        pytest.param("NEXTEXT_SERVICE_TOKENS", "docint", id="malformed-tokens"),
    ],
)
def test_startup_fails_fast_on_malformed_identity_config(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    """The backend refuses to start rather than serve with a broken identity config.

    Args:
        monkeypatch: Pytest environment patcher.
        name: The variable to set.
        value: Its malformed value.
    """
    monkeypatch.setenv(name, value)
    with pytest.raises(PrincipalConfigError), TestClient(create_app()):
        pass
