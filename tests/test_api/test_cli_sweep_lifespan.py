"""The backend sweeps what CLI runs left, at startup and daily."""

from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient

from nextext.api import cli_files
from nextext.api import main as main_module
from nextext.api.main import create_app
from tests.audit_helpers import AuditCapture


def test_startup_sweeps_old_cli_runs(monkeypatch: pytest.MonkeyPatch, audit_capture: AuditCapture) -> None:
    """A run older than the period is deleted when the backend starts, recorded, and the config says so."""
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    root = cli_files.CLI_ROOT
    old = root / "output" / "old-run"
    old.mkdir(parents=True)
    moment = time.time() - 400 * 86_400
    os.utime(old, (moment, moment))
    (root / ".retention-since").write_text(f"{int(moment)} 365\n", encoding="utf-8")
    with TestClient(create_app(audit_emitter_factory=audit_capture.factory)):
        pass
    assert not old.exists()
    (event,) = audit_capture.of("delete", "cli.sweep")
    assert event["target"]["count"] == 1
    (config,) = audit_capture.of("admin", "config.audit")
    assert (config["criteria"]["cli_retention_days"], config["criteria"]["retention_notice_days"]) == (365, 30)


def test_a_bad_period_stops_startup(monkeypatch: pytest.MonkeyPatch, audit_capture: AuditCapture) -> None:
    """NEXTEXT_CLI_RETENTION_DAYS that is no period refuses to start the backend."""
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    monkeypatch.setenv("NEXTEXT_CLI_RETENTION_DAYS", "a year")
    with pytest.raises(ValueError, match="NEXTEXT_CLI_RETENTION_DAYS"):
        with TestClient(create_app(audit_emitter_factory=audit_capture.factory)):
            pass
