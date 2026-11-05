"""Tests for the sweep of what ``make cli`` leaves on the scratch volume."""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import pytest

from nextext.api.cli_files import CliFileSweeper, entry_reference
from nextext.utils.audit_event import AuditEmitter, AuditWriteError
from tests.audit_helpers import AuditCapture

DAY = 86_400
NOW = 1_793_880_000.0


def _age(path: Path, days: float) -> None:
    """Set a path's modification time ``days`` before :data:`NOW`.

    Args:
        path: The file or directory.
        days: Its age.
    """
    moment = NOW - days * DAY
    os.utime(path, (moment, moment), follow_symlinks=False)


def _run(root: Path, name: str, days: float) -> Path:
    """Leave a run's output directory with one file, ``days`` old.

    Args:
        root: The CLI root.
        name: The run's stem.
        days: Its age.

    Returns:
        The directory.
    """
    run = root / "output" / name
    run.mkdir(parents=True)
    (run / "summary.txt").write_text("zz-sentinel-content", encoding="utf-8")
    _age(run / "summary.txt", days)
    _age(run, days)
    return run


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path]:
    """Create a CLI root and a legacy output directory.

    Args:
        tmp_path: Temporary directory.

    Returns:
        The root and the legacy directory.
    """
    root, legacy = tmp_path / "nextext-cli", tmp_path / "output"
    root.mkdir()
    return root, legacy


def _sweeper(root: Path, legacy: Path, *, days: int | None = 30, since_days: float | None = 400) -> CliFileSweeper:
    """Build a sweeper whose period took effect ``since_days`` before :data:`NOW`.

    Args:
        root: The CLI root.
        legacy: The legacy output directory.
        days: The period.
        since_days: When it took effect; ``None`` leaves no state.

    Returns:
        The sweeper.
    """
    if since_days is not None and days is not None:
        (root / ".retention-since").write_text(f"{int(NOW - since_days * DAY)} {days}\n", encoding="utf-8")
    return CliFileSweeper(root, legacy, days=days, notice_days=30)


def _emitter() -> tuple[AuditEmitter, AuditCapture]:
    capture = AuditCapture()
    return capture.started(), capture


def test_expired_run_directory_is_removed_and_recorded(roots: tuple[Path, Path]) -> None:
    """A run older than the period goes, recorded first, by a hash of its path, never its name."""
    root, legacy = roots
    old = _run(root, "interview-alice", 31)
    young = _run(root, "interview-bob", 10)
    emitter, capture = _emitter()
    result = _sweeper(root, legacy).sweep(NOW, emitter)
    assert result.removed == ["output/interview-alice"]
    assert not old.exists()
    assert young.exists()
    (event,) = capture.of("delete", "cli.sweep")
    assert event["actor"] == {"user": None, "via": "system"}
    assert event["target"] == {"kind": "cli_file", "ids": [entry_reference("output/interview-alice")], "count": 1}
    assert event["criteria"]["period_days"] == 30
    assert "interview" not in capture.stream.getvalue()
    assert "zz-sentinel-content" not in capture.stream.getvalue()


def test_entry_reference_is_a_hash_of_the_path() -> None:
    """References are ``path-sha256:`` plus the SHA-256 of the entry's path below the scratch volume."""
    digest = hashlib.sha256(b"output/x").hexdigest()
    assert entry_reference("output/x") == f"path-sha256:{digest}"


def test_young_run_stays(roots: tuple[Path, Path]) -> None:
    """Nothing within the period is touched, and nothing is recorded."""
    root, legacy = roots
    young = _run(root, "run", 29)
    emitter, capture = _emitter()
    result = _sweeper(root, legacy).sweep(NOW, emitter)
    assert (result.removed, young.exists()) == ([], True)
    assert capture.of("delete") == []


def test_a_run_still_writing_is_young(roots: tuple[Path, Path]) -> None:
    """An old directory with a file written just now is young: its newest time decides."""
    root, legacy = roots
    run = _run(root, "long-run", 400)
    (run / "transcript.tsv").write_text("x", encoding="utf-8")
    _age(run / "transcript.tsv", 0)
    _age(run, 400)
    emitter, _ = _emitter()
    assert _sweeper(root, legacy).sweep(NOW, emitter).removed == []
    assert run.exists()


def test_inputs_in_the_root_are_swept_too(roots: tuple[Path, Path]) -> None:
    """An input copied into the root goes by its own age; the state file never does."""
    root, legacy = roots
    clip = root / "clip.mp4"
    clip.write_bytes(b"x")
    _age(clip, 40)
    emitter, _ = _emitter()
    result = _sweeper(root, legacy).sweep(NOW, emitter)
    assert result.removed == ["clip.mp4"]
    assert (root / ".retention-since").exists()


def test_legacy_output_is_swept_too(roots: tuple[Path, Path]) -> None:
    """Outputs earlier releases left under /tmp/output follow the same rule."""
    root, legacy = roots
    legacy.mkdir()
    old_legacy, young_legacy = legacy / "old-run", legacy / "young-run"
    for path, days in ((old_legacy, 50), (young_legacy, 2)):
        path.mkdir()
        _age(path, days)
    emitter, _ = _emitter()
    result = _sweeper(root, legacy).sweep(NOW, emitter)
    assert result.removed == ["legacy:old-run"]
    assert not old_legacy.exists()
    assert young_legacy.exists()


def test_grace_before_the_rule_took_effect(roots: tuple[Path, Path]) -> None:
    """A period seen for the first time deletes nothing for the notice period."""
    root, legacy = roots
    old = _run(root, "old", 400)
    emitter, _ = _emitter()
    sweeper = _sweeper(root, legacy, since_days=None)
    assert sweeper.sweep(NOW, emitter).removed == []
    assert (root / ".retention-since").read_text(encoding="utf-8").split() == [str(int(NOW)), "30"]
    assert sweeper.sweep(NOW + 29 * DAY, emitter).removed == []
    assert old.exists()
    assert sweeper.sweep(NOW + 30 * DAY, emitter).removed == ["output/old"]


def test_a_changed_period_restarts_the_grace(roots: tuple[Path, Path]) -> None:
    """Lowering the period deletes nothing more before its notice has passed."""
    root, legacy = roots
    (root / ".retention-since").write_text(f"{int(NOW - 400 * DAY)} 365\n", encoding="utf-8")
    run = _run(root, "run", 40)
    emitter, _ = _emitter()
    assert CliFileSweeper(root, legacy, days=30, notice_days=30).sweep(NOW, emitter).removed == []
    assert run.exists()


def test_symlink_is_not_followed(roots: tuple[Path, Path], tmp_path: Path) -> None:
    """A link in the root is removed itself; its target stays."""
    root, legacy = roots
    target = tmp_path / "elsewhere.txt"
    target.write_text("keep", encoding="utf-8")
    link = root / "link.txt"
    link.symlink_to(target)
    _age(link, 400)
    emitter, _ = _emitter()
    _sweeper(root, legacy).sweep(NOW, emitter)
    assert target.exists()
    assert not link.is_symlink()


def test_job_directory_is_never_touched(roots: tuple[Path, Path]) -> None:
    """The sweep reads only its own root and the legacy output, never the job manager's directory."""
    root, legacy = roots
    jobs = root.parent / "nextext-jobs" / "job-old"
    jobs.mkdir(parents=True)
    _age(jobs, 400)
    emitter, _ = _emitter()
    _sweeper(root, legacy).sweep(NOW, emitter)
    assert jobs.exists()


def test_record_failure_deletes_nothing(roots: tuple[Path, Path]) -> None:
    """When the record cannot be written, the run stays."""
    root, legacy = roots
    run = _run(root, "old", 400)

    class Broken(io.StringIO):
        def write(self, s: str) -> int:
            if '"delete"' in s:
                raise OSError("log pipe closed")
            return super().write(s)

    emitter = AuditEmitter("nextext", "0.0.0+test", stream=Broken(), heartbeat_interval_s=0)
    emitter.start()
    with pytest.raises(AuditWriteError):
        _sweeper(root, legacy).sweep(NOW, emitter)
    assert run.exists()


def test_undeletable_entry_is_recorded_as_error(roots: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """An entry that cannot be removed is recorded with outcome error; the others still go."""
    root, legacy = roots
    _run(root, "stuck", 400)
    _run(root, "fine", 400)
    real = CliFileSweeper._remove

    def remove(self: CliFileSweeper, path: Path) -> None:
        if path.name == "stuck":
            raise PermissionError("denied")
        real(self, path)

    monkeypatch.setattr(CliFileSweeper, "_remove", remove)
    emitter, capture = _emitter()
    result = _sweeper(root, legacy).sweep(NOW, emitter)
    assert (result.removed, result.failed) == (["output/fine"], ["output/stuck"])
    outcomes = [event["outcome"] for event in capture.of("delete", "cli.sweep")]
    assert outcomes == ["ok", "error"]


def test_off_does_nothing(roots: tuple[Path, Path]) -> None:
    """With retention off nothing is scanned, written or deleted."""
    root, legacy = roots
    run = _run(root, "old", 4000)
    emitter, capture = _emitter()
    assert CliFileSweeper(root, legacy, days=None, notice_days=30).sweep(NOW, emitter).removed == []
    assert run.exists()
    assert not (root / ".retention-since").exists()
    assert capture.of("delete") == []


def test_missing_root_is_created_for_the_state(tmp_path: Path) -> None:
    """On a fresh volume the root does not exist yet: the sweep creates it, deletes nothing."""
    root, legacy = tmp_path / "nextext-cli", tmp_path / "output"
    emitter, _ = _emitter()
    assert CliFileSweeper(root, legacy, days=30, notice_days=30).sweep(NOW, emitter).removed == []
    assert (root / ".retention-since").exists()
