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
    monkeypatch.setattr(cli_files, "_stored_at", lambda path: os.lstat(path).st_mtime)
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


def test_startup_counts_legacy_inputs_in_tmp(monkeypatch: pytest.MonkeyPatch, audit_capture: AuditCapture) -> None:
    """Inputs earlier releases left directly in /tmp are counted in the log (never named), not deleted."""
    from loguru import logger

    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    (cli_files.SCRATCH / "interview-old.wav").write_bytes(b"x")
    lines: list[str] = []
    sink = logger.add(lines.append, level="WARNING")
    try:
        with TestClient(create_app(audit_emitter_factory=audit_capture.factory)):
            pass
    finally:
        logger.remove(sink)
    assert any("1 file(s) left directly in /tmp" in line and "make cli-clean LEGACY=1" in line for line in lines)
    assert not any("interview-old" in line for line in lines)
    assert (cli_files.SCRATCH / "interview-old.wav").exists()


def test_a_bad_period_stops_startup(monkeypatch: pytest.MonkeyPatch, audit_capture: AuditCapture) -> None:
    """NEXTEXT_CLI_RETENTION_DAYS that is no period refuses to start the backend."""
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    monkeypatch.setenv("NEXTEXT_CLI_RETENTION_DAYS", "a year")
    with pytest.raises(ValueError, match="NEXTEXT_CLI_RETENTION_DAYS"):
        with TestClient(create_app(audit_emitter_factory=audit_capture.factory)):
            pass


def test_switching_off_forgets_when_the_period_took_effect(
    monkeypatch: pytest.MonkeyPatch, audit_capture: AuditCapture
) -> None:
    """With retention off the state goes, so switching on again gives a fresh notice."""
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    monkeypatch.setenv("NEXTEXT_CLI_RETENTION_DAYS", "off")
    root = cli_files.CLI_ROOT
    root.mkdir(parents=True, exist_ok=True)
    (root / ".retention-since").write_text(f"{int(time.time()) - 400 * 86_400} 30\n", encoding="utf-8")
    with TestClient(create_app(audit_emitter_factory=audit_capture.factory)):
        pass
    assert not (root / ".retention-since").exists()


def test_shutdown_waits_for_a_running_sweep(monkeypatch: pytest.MonkeyPatch, audit_capture: AuditCapture) -> None:
    """A delete in progress finishes before the job manager stops and the audit chain closes."""
    from nextext.api.jobs import JobManager

    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    monkeypatch.setattr(main_module, "CLI_SWEEP_EVERY_S", 0.0)
    order: list[str] = []
    calls = {"n": 0}

    def sweep(sweeper: object, emitter: object) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            time.sleep(0.3)
            order.append("sweep finished")

    real_stop = JobManager.stop

    async def stop(self: JobManager) -> None:
        order.append("job manager stopped")
        await real_stop(self)

    monkeypatch.setattr(main_module, "_sweep_cli_files", sweep)
    monkeypatch.setattr(JobManager, "stop", stop)
    with TestClient(create_app(audit_emitter_factory=audit_capture.factory)):
        while calls["n"] < 2:
            time.sleep(0.01)
    assert order == ["sweep finished", "job manager stopped"]


def test_a_failing_startup_sweep_still_stops_the_job_manager(
    monkeypatch: pytest.MonkeyPatch, audit_capture: AuditCapture
) -> None:
    """When the startup sweep cannot record, startup fails and the job manager is stopped anyway."""
    from nextext.api.jobs import JobManager
    from nextext.utils.audit_event import AuditWriteError

    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    stopped: list[bool] = []
    real_stop = JobManager.stop

    async def stop(self: JobManager) -> None:
        stopped.append(True)
        await real_stop(self)

    def sweep(sweeper: object, emitter: object) -> None:
        raise AuditWriteError("trail unwritable")

    monkeypatch.setattr(main_module, "_sweep_cli_files", sweep)
    monkeypatch.setattr(JobManager, "stop", stop)
    with pytest.raises(AuditWriteError):
        with TestClient(create_app(audit_emitter_factory=audit_capture.factory)):
            pass
    assert stopped == [True]


def test_the_daily_sweep_survives_an_unexpected_error(
    monkeypatch: pytest.MonkeyPatch, audit_capture: AuditCapture
) -> None:
    """One failing sweep is logged; the next still runs."""
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    monkeypatch.setattr(main_module, "CLI_SWEEP_EVERY_S", 0.0)
    calls = {"n": 0}

    def sweep(sweeper: object, emitter: object) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("disk gone")

    monkeypatch.setattr(main_module, "_sweep_cli_files", sweep)
    with TestClient(create_app(audit_emitter_factory=audit_capture.factory)):
        deadline = time.monotonic() + 2
        while calls["n"] < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
    assert calls["n"] >= 3


def test_a_state_that_cannot_be_forgotten_never_stops_startup(
    monkeypatch: pytest.MonkeyPatch, audit_capture: AuditCapture
) -> None:
    """Switching retention off on a volume where the state cannot be removed logs it and starts anyway."""
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    monkeypatch.setenv("NEXTEXT_CLI_RETENTION_DAYS", "off")

    def refuse(root: object) -> None:
        raise PermissionError("read-only")

    monkeypatch.setattr(cli_files, "forget_rule", refuse)
    with TestClient(create_app(audit_emitter_factory=audit_capture.factory)):
        pass


def test_a_sweep_failing_at_shutdown_is_logged(monkeypatch: pytest.MonkeyPatch, audit_capture: AuditCapture) -> None:
    """A sweep that fails while shutdown waits for it leaves a log line, not silence."""
    from loguru import logger

    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    monkeypatch.setattr(main_module, "CLI_SWEEP_EVERY_S", 0.0)
    calls = {"n": 0}

    def sweep(sweeper: object, emitter: object) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            time.sleep(0.2)
            raise RuntimeError("disk gone")

    monkeypatch.setattr(main_module, "_sweep_cli_files", sweep)
    lines: list[str] = []
    sink = logger.add(lines.append, level="ERROR")
    try:
        with TestClient(create_app(audit_emitter_factory=audit_capture.factory)):
            while calls["n"] < 2:
                time.sleep(0.01)
    finally:
        logger.remove(sink)
    assert any("sweep" in line.lower() and "shutdown" in line.lower() for line in lines)
