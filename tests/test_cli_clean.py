"""Tests for ``nextext-cli-clean`` (``make cli-clean``): delete what CLI runs left, now, recorded."""

from __future__ import annotations

from pathlib import Path

import pytest

from nextext import cli_clean
from nextext.api import cli_files
from nextext.api.cli_files import entry_reference
from tests.audit_helpers import AuditCapture


@pytest.fixture
def scratch(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Leave two runs, an input, and legacy files on the throwaway scratch volume.

    Args:
        monkeypatch: Sets the operator.

    Returns:
        The scratch volume.
    """
    monkeypatch.setenv("AUDIT_CLI_USER", "ops.example")
    root = cli_files.CLI_ROOT
    for run in ("run-a", "run-b"):
        (root / "output" / run).mkdir(parents=True)
        (root / "output" / run / "summary.txt").write_text("x", encoding="utf-8")
    (root / "clip.mp4").write_bytes(b"x")
    base = cli_files.SCRATCH
    (base / "old-input.wav").write_bytes(b"x")
    (base / "audit-stream.lock").write_bytes(b"")
    (base / ".hidden").write_bytes(b"x")
    (base / "nextext-jobs").mkdir()
    return base


def test_clean_all(scratch: Path) -> None:
    """Without options every run and input in the CLI's directory goes, recorded with the operator."""
    capture = AuditCapture()
    assert cli_clean.main([], emitter_factory=capture.factory) == 0
    root = cli_files.CLI_ROOT
    assert not (root / "output" / "run-a").exists()
    assert not (root / "clip.mp4").exists()
    assert (scratch / "old-input.wav").exists()
    (event,) = capture.of("delete", "cli.clean")
    assert event["actor"] == {"user": "ops.example", "via": "cli"}
    assert event["target"]["count"] == 3
    assert "run-a" not in capture.stream.getvalue()


def test_clean_one_run(scratch: Path) -> None:
    """``--run NAME`` deletes that entry only."""
    capture = AuditCapture()
    assert cli_clean.main(["--run", "output/run-a"], emitter_factory=capture.factory) == 0
    assert not (cli_files.CLI_ROOT / "output" / "run-a").exists()
    assert (cli_files.CLI_ROOT / "output" / "run-b").exists()
    (event,) = capture.of("delete", "cli.clean")
    assert event["target"]["ids"] == [entry_reference("output/run-a")]


def test_legacy_keeps_the_lock_file_and_dotfiles(scratch: Path) -> None:
    """``--legacy`` also removes regular files directly in /tmp, never the audit lock, dotfiles or directories."""
    capture = AuditCapture()
    assert cli_clean.main(["--legacy"], emitter_factory=capture.factory) == 0
    assert not (scratch / "old-input.wav").exists()
    assert (scratch / "audit-stream.lock").exists()
    assert (scratch / ".hidden").exists()
    assert (scratch / "nextext-jobs").is_dir()
    (event,) = capture.of("delete", "cli.clean")
    assert entry_reference("tmp:old-input.wav") in event["target"]["ids"]


def test_unknown_run_exits_1_and_records_nothing(scratch: Path) -> None:
    """A name that is not an entry: exit 1, nothing deleted, nothing recorded."""
    capture = AuditCapture()
    assert cli_clean.main(["--run", "output/nope"], emitter_factory=capture.factory) == 1
    assert capture.of("delete") == []
    assert (cli_files.CLI_ROOT / "output" / "run-a").exists()


def test_nothing_to_clean_records_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty scratch volume: exit 0, no record."""
    capture = AuditCapture()
    assert cli_clean.main([], emitter_factory=capture.factory) == 0
    assert capture.of("delete") == []


def test_dry_run_lists_counts_and_deletes_nothing(scratch: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``--dry-run`` says how many entries (and legacy inputs) there are, deletes and records nothing."""
    capture = AuditCapture()
    assert cli_clean.main(["--dry-run", "--legacy"], emitter_factory=capture.factory) == 0
    err = capsys.readouterr().err
    assert "would delete 4 CLI file(s), 1 of them left directly in /tmp" in err
    assert (scratch / "old-input.wav").exists()
    assert (cli_files.CLI_ROOT / "clip.mp4").exists()
    assert capture.events == []
    assert "old-input" not in err
