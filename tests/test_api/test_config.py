"""Tests for ``GET /api/v1/config``."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nextext.api import main as main_module
from nextext.api.main import create_app
from tests.audit_helpers import AuditCapture


def test_get_config_language(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /api/v1/config reports the resolved UI language."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "de")

    response = api_client.get("/api/v1/config")

    assert response.status_code == 200
    assert response.json() == {"language": "de", "require_purpose": [], "faces": "off"}


@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        pytest.param(None, [], id="unset"),
        pytest.param("", [], id="empty"),
        pytest.param("query,disclose", ["disclose", "query"], id="sorted"),
    ],
)
def test_config_lists_the_actions_that_need_a_purpose(
    monkeypatch: pytest.MonkeyPatch, audit_capture: AuditCapture, setting: str | None, expected: list[str]
) -> None:
    """The SPA reads which actions need a stated purpose from /config, which itself is not recorded.

    Args:
        monkeypatch: Pytest patcher.
        audit_capture: Where the app's audit events go.
        setting: The ``AUDIT_REQUIRE_PURPOSE`` value, or ``None`` for unset.
        expected: The list /config reports.
    """
    if setting is not None:
        monkeypatch.setenv("AUDIT_REQUIRE_PURPOSE", setting)
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    with TestClient(create_app(audit_emitter_factory=audit_capture.factory)) as client:
        before = len(audit_capture.events)
        response = client.get("/api/v1/config")
        assert len(audit_capture.events) == before
    assert response.status_code == 200
    assert response.json()["require_purpose"] == expected
    audit_capture.assert_chain_ok()
