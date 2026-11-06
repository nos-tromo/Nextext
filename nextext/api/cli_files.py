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

Its age is the later of its modification and change times (``docker compose
cp`` keeps a source's old mtime, but the change time is when it reached the
volume and cannot be set back); a directory's is the newest anywhere below it,
so a run that is still writing is young. Symbolic links are never
followed: a link is removed itself. ``/tmp/nextext-jobs`` (the job manager's)
and anything else in ``/tmp`` is never touched.

A period that is new deletes nothing for ``RETENTION_NOTICE_DAYS`` after it
took effect. A changed period never deletes sooner than announced: a longer
one applies at once (also to what was due under the shorter one but not yet
deleted), and a shorter one gives the notice period only to what it makes due
sooner while the rule in force keeps deleting. Changing the notice alone moves
nothing. The state is ``/tmp/nextext-cli/.retention-since``: ``<since> <days>
<grace until> [<days in force> <their grace until>]`` in epoch seconds (the
``<since> <days>`` of the release before is read as a grace that started at
``<since>``). Switching retention off forgets it. Every deletion is recorded first (``delete`` /
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
from nextext.utils.audit_event import Actor, AuditEmitter, AuditWriteError

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
        waiting_until: When the period takes full effect, while its notice runs.
    """

    removed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    waiting_until: float | None = None


def stored_at(path: Path) -> float:
    """Return when a path was last written to the volume: the later of its mtime and ctime (not following links).

    Args:
        path: The path.

    Returns:
        float: Seconds since the epoch.
    """
    info = os.lstat(path)
    return max(info.st_mtime, info.st_ctime)


#: The clock :func:`_newest` reads; tests that backdate mtimes replace it.
_stored_at = stored_at


def _newest(path: Path) -> float:
    """Return the newest modification time of a path and, for a directory, of everything below it.

    Args:
        path: The entry.

    Returns:
        float: Seconds since the epoch.
    """
    newest = _stored_at(path)
    if path.is_dir() and not path.is_symlink():
        for directory, dirs, files in os.walk(path, followlinks=False):
            for name in (*dirs, *files):
                try:
                    newest = max(newest, _stored_at(Path(directory, name)))
                except FileNotFoundError:
                    continue
    return newest


@dataclass(frozen=True)
class RetentionRule:
    """When what CLI runs leave may be deleted.

    Attributes:
        days: The period.
        grace_until: Until when the period does not delete on its own.
        in_force_days: The period in force before a change, deleting meanwhile.
        in_force_grace: Since when that period deletes.
    """

    days: int
    grace_until: float
    in_force_days: int | None = None
    in_force_grace: float | None = None

    def deletable(self, written: float, now: float) -> bool:
        """Whether something stored at ``written`` may go at ``now``.

        Args:
            written: When it was stored.
            now: The current time.

        Returns:
            bool: ``True`` once the period has passed and the rule allows it.
        """
        if written + self.days * 86_400 > now:
            return False
        if now >= self.grace_until:
            return True
        return (
            self.in_force_days is not None
            and self.in_force_grace is not None
            and now >= self.in_force_grace
            and written + self.in_force_days * 86_400 <= now
        )

    def deadline(self, written: float) -> float:
        """When something stored at ``written`` is due.

        Args:
            written: When it was stored.

        Returns:
            float: Seconds since the epoch.
        """
        own = written + self.days * 86_400
        if self.in_force_days is None or self.in_force_grace is None:
            return max(own, self.grace_until)
        in_force = max(written + self.in_force_days * 86_400, self.in_force_grace)
        return max(own, min(self.grace_until, in_force))


#: The longest notice a state may record, in days (as ``RETENTION_NOTICE_DAYS``).
_MAX_NOTICE_DAYS: Final = 365


def _parse_state(text: str, notice_days: int) -> tuple[float, int, float, int | None, float | None, bool]:
    """Read and check the state file's fields.

    Args:
        text: Its content.
        notice_days: The notice, for the two-field form of the release before.

    Returns:
        The time it took effect, the period, the grace end, the rule in force,
        and whether the state is the release before's two-field form.

    Raises:
        ValueError: If it holds no time and period, or impossible values.
    """
    parts = text.split()
    if len(parts) not in (2, 3, 5) or not all(part.isdigit() and len(part) <= 11 for part in parts):
        raise ValueError("the CLI retention state holds no time and period")
    since, days = float(parts[0]), int(parts[1])
    legacy = len(parts) == 2
    grace = since + notice_days * 86_400 if legacy else float(parts[2])
    in_force, in_force_grace = (int(parts[3]), float(parts[4])) if len(parts) == 5 else (None, None)
    if days < 1 or not since <= grace <= since + _MAX_NOTICE_DAYS * 86_400:
        raise ValueError("the CLI retention state holds impossible values")
    if in_force is not None and in_force_grace is not None and (in_force < 1 or in_force_grace > grace):
        raise ValueError("the CLI retention state holds an impossible period in force")
    return since, days, grace, in_force, in_force_grace, legacy


def _write_state(root: Path, since: float, rule: RetentionRule) -> None:
    """Write the state (a temporary file renamed into place).

    Args:
        root: The CLI root.
        since: When the period took effect.
        rule: The rule.
    """
    root.mkdir(parents=True, exist_ok=True)
    fields = [int(since), rule.days, int(rule.grace_until)]
    if rule.in_force_days is not None and rule.in_force_grace is not None:
        fields += [rule.in_force_days, int(rule.in_force_grace)]
    state = root / STATE_NAME
    temp = state.with_name(f"{STATE_NAME}.tmp")
    temp.write_text(" ".join(str(field) for field in fields) + "\n", encoding="utf-8")
    temp.replace(state)


def read_rule(root: Path, *, days: int, notice_days: int, now: float, record: bool = False) -> RetentionRule:
    """Return the rule in force, moving to ``days`` when it is new or changed.

    A longer period applies at once (within a running first grace, at its
    end). A shorter one gives the notice only to what it makes due sooner,
    never ending before a grace already running; meanwhile the rule actually
    in force before the change still deletes.

    Args:
        root: The CLI root holding the state.
        days: The configured period.
        notice_days: The configured notice.
        now: The current time.
        record: Write a new, changed or upgraded state; without it the state is only read.

    Returns:
        RetentionRule: The rule.

    Raises:
        OSError: If the state cannot be read or written.
        ValueError: If the state file is damaged.
    """
    state = root / STATE_NAME
    fresh = RetentionRule(days, now + notice_days * 86_400)
    if not state.exists():
        rule, since = fresh, now
    else:
        since, recorded, grace, in_force, in_force_grace, legacy = _parse_state(
            state.read_text(encoding="utf-8"), notice_days
        )
        dated_ahead = since > now
        if dated_ahead:
            # A clock set back: keep the rule, its dates never more than a notice ahead.
            logger.warning("The CLI retention state is dated in the future; its dates are capped at now.")
            since, grace = now, min(grace, fresh.grace_until)
            if in_force_grace is not None:
                in_force_grace = min(in_force_grace, grace)
        if recorded == days:
            rule = RetentionRule(days, grace, in_force, in_force_grace)
            if (legacy or dated_ahead) and record:
                # Fixed now, so a later change of the notice moves nothing.
                _write_state(root, since, rule)
            return rule
        else:
            if now >= grace or in_force is None or in_force_grace is None:
                effective, effective_grace = recorded, grace
            else:
                effective, effective_grace = max(in_force, recorded), in_force_grace
            since = now
            if days >= effective:
                rule = RetentionRule(days, max(now, effective_grace))
            else:
                running = grace if grace > now else now
                rule = RetentionRule(days, max(fresh.grace_until, running), effective, effective_grace)
    if record:
        _write_state(root, since, rule)
    return rule


def forget_rule(root: Path) -> None:
    """Forget the state (retention switched off), so switching on gives a fresh notice.

    Args:
        root: The CLI root.
    """
    (root / STATE_NAME).unlink(missing_ok=True)


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
            notice_days: How long a new period deletes nothing (and a changed one nothing sooner).
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
        if self.root.is_dir() and not self.root.is_symlink():
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
        rule = read_rule(self.root, days=self.days, notice_days=self.notice_days, now=now, record=True)
        if now < rule.grace_until:
            result.waiting_until = rule.grace_until
        due = [entry for entry in self.scan(now) if rule.deletable(entry.written_at, now)]
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
            try:
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
            except AuditWriteError:
                logger.error(
                    "{} deleted, {} could not be; the failure record could not be written.",
                    len(result.removed),
                    len(result.failed),
                )
                return result
        logger.info("Deleted {} CLI run file(s) past the retention period.", len(result.removed))
        return result
