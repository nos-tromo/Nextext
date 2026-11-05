"""The sweep of what ``make cli`` leaves on the scratch volume.

``make cli`` runs ``nextext-cli`` inside the backend with ``/tmp/nextext-cli``
(on the ``tmp-jobs`` volume) as its working directory: the operator copies the
input there and the outputs land under ``output/<stem>/``. Nothing in the
pipeline deletes them. This module does, once they are older than the
retention period (``NEXTEXT_CLI_RETENTION_DAYS``): the backend sweeps at
startup and then daily, and ``make cli-clean`` deletes on demand
(:mod:`nextext.cli_clean`).

An entry is one of:

* a file or directory directly in ``/tmp/nextext-cli`` (an input), except
  ``output/`` and the state file;
* a run's directory in ``/tmp/nextext-cli/output/``;
* a run's directory in ``/tmp/output/``, where releases before this one wrote.

Its age is its own modification time; a directory's is the newest anywhere
below it, so a run that is still writing is young. Symbolic links are never
followed: a link is removed itself. ``/tmp/nextext-jobs`` (the job manager's)
and anything else in ``/tmp`` is never touched.

A period that is new or changed deletes nothing for ``RETENTION_NOTICE_DAYS``
after it took effect, kept in ``/tmp/nextext-cli/.retention-since``
(``<epoch> <days>``). Every deletion is recorded first (``delete`` /
``cli.sweep``, by the system), naming each entry by the SHA-256 of its path
below the scratch volume, never by name: a file name can name a person. An
entry that cannot be removed is recorded again with ``outcome="error"``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from loguru import logger

from nextext.audit import emit_with_ids
from nextext.utils.audit_event import Actor, AuditEmitter

#: The scratch volume (``tmp-jobs``), the backend's ``/tmp``.
SCRATCH: Final = Path("/tmp")
#: Where ``make cli`` runs.
CLI_ROOT: Final = Path("/tmp/nextext-cli")
#: Where releases before this one wrote the CLI's outputs.
LEGACY_OUTPUT: Final = Path("/tmp/output")
#: The state file, in the root.
STATE_NAME: Final = ".retention-since"
SYSTEM: Final = Actor(None, "system")


def entry_reference(relative: str) -> str:
    """Name an entry in a record without its name.

    Args:
        relative: Its path below the scratch volume, e.g. ``output/<stem>``.

    Returns:
        str: ``path-sha256:<hex>``.
    """
    return f"path-sha256:{hashlib.sha256(relative.encode()).hexdigest()}"


@dataclass(frozen=True)
class CliEntry:
    """One thing a CLI run left.

    Attributes:
        name: Its path below the scratch volume (``legacy:<stem>`` for /tmp/output).
        path: Where it is.
        written_at: Its newest modification time.
        due_at: When it is due.
    """

    name: str
    path: Path
    written_at: float
    due_at: float


@dataclass
class CliSweep:
    """What one sweep did.

    Attributes:
        removed: The entries deleted, by name.
        failed: The entries that could not be deleted.
        waiting_until: When deletion starts, while the period's notice runs.
    """

    removed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    waiting_until: float | None = None


def _newest(path: Path) -> float:
    """Return the newest modification time of a path and, for a directory, of everything below it.

    Args:
        path: The entry.

    Returns:
        float: Seconds since the epoch.
    """
    newest = os.lstat(path).st_mtime
    if path.is_dir() and not path.is_symlink():
        for directory, dirs, files in os.walk(path, followlinks=False):
            for name in (*dirs, *files):
                try:
                    newest = max(newest, os.lstat(os.path.join(directory, name)).st_mtime)
                except FileNotFoundError:
                    continue
    return newest


class CliFileSweeper:
    """Deletes what CLI runs left once the retention period has passed."""

    def __init__(
        self,
        root: Path = CLI_ROOT,
        legacy: Path = LEGACY_OUTPUT,
        *,
        days: int | None,
        notice_days: int,
    ) -> None:
        """Prepare a sweeper; nothing is read until :meth:`sweep` or :meth:`scan`.

        Args:
            root: The CLI's working directory.
            legacy: The output directory of earlier releases.
            days: The period; ``None`` is off.
            notice_days: How long a new or changed period deletes nothing.
        """
        self.root = root
        self.legacy = legacy
        self.days = days
        self.notice_days = notice_days

    def scan(self, now: float) -> list[CliEntry]:
        """List every entry with its age and due time.

        Args:
            now: The current time (used only for entries' due times).

        Returns:
            list[CliEntry]: Sorted by name.
        """
        period = (self.days or 0) * 86_400
        candidates: list[tuple[str, Path]] = []
        if self.root.is_dir():
            for child in self.root.iterdir():
                if child.name in {STATE_NAME, "output"}:
                    continue
                candidates.append((child.name, child))
            output = self.root / "output"
            if output.is_dir() and not output.is_symlink():
                candidates.extend((f"output/{run.name}", run) for run in output.iterdir())
        if self.legacy.is_dir() and not self.legacy.is_symlink():
            candidates.extend((f"legacy:{run.name}", run) for run in self.legacy.iterdir())
        entries = []
        for name, path in candidates:
            try:
                written = _newest(path)
            except FileNotFoundError:
                continue
            entries.append(CliEntry(name, path, written, written + period))
        return sorted(entries, key=lambda entry: entry.name)

    def _since(self, now: float) -> float:
        """Return when the period in force took effect, recording ``now`` for a new or changed one.

        Args:
            now: The current time.

        Returns:
            float: Seconds since the epoch.

        Raises:
            OSError: If the state cannot be read or written.
            ValueError: If the state file holds no time and period.
        """
        state = self.root / STATE_NAME
        if state.exists():
            parts = state.read_text(encoding="utf-8").split()
            if len(parts) != 2 or not all(part.isdigit() for part in parts):
                raise ValueError(f"{state} holds no time and period")
            since, recorded = int(parts[0]), int(parts[1])
            if recorded == self.days:
                return float(since)
        self.root.mkdir(parents=True, exist_ok=True)
        temp = state.with_name(f"{STATE_NAME}.tmp")
        temp.write_text(f"{int(now)} {self.days}\n", encoding="utf-8")
        temp.replace(state)
        return now

    def _remove(self, path: Path) -> None:
        """Remove one entry, never following a link.

        Args:
            path: The entry.

        Raises:
            OSError: If it cannot be removed.
        """
        if path.is_symlink() or not path.is_dir():
            path.unlink(missing_ok=True)
        else:
            shutil.rmtree(path)

    def sweep(self, now: float, emitter: AuditEmitter) -> CliSweep:
        """Delete every entry past its due time, once the period's notice has passed.

        Args:
            now: The sweep's time.
            emitter: The backend's started audit emitter.

        Returns:
            CliSweep: What was deleted.

        Raises:
            AuditWriteError: If the record cannot be written; nothing was deleted.
            OSError: If the state cannot be read or written.
            ValueError: If the state file is damaged.
        """
        result = CliSweep()
        if self.days is None:
            return result
        since = self._since(now)
        start = since + self.notice_days * 86_400
        if now < start:
            result.waiting_until = start
            return result
        due = [entry for entry in self.scan(now) if entry.due_at <= now]
        if not due:
            return result
        criteria = {"period_days": self.days, "trigger": "schedule"}
        emit_with_ids(
            emitter,
            "delete",
            "cli.sweep",
            SYSTEM,
            kind="cli_file",
            ids=[entry_reference(entry.name) for entry in due],
            criteria=criteria,
        )
        for entry in due:
            try:
                self._remove(entry.path)
            except OSError as exc:
                logger.warning("Could not delete a CLI run's file ({}).", type(exc).__name__)
                result.failed.append(entry.name)
            else:
                result.removed.append(entry.name)
        if result.failed:
            emit_with_ids(
                emitter,
                "delete",
                "cli.sweep",
                SYSTEM,
                kind="cli_file",
                ids=[entry_reference(name) for name in result.failed],
                outcome="error",
                criteria=criteria,
            )
        logger.info("Deleted {} CLI run file(s) past the retention period.", len(result.removed))
        return result
