"""Durable job storage for the Nextext FastAPI backend.

Persistence is opt-in per job (``JobOptions.persist``). Ephemeral jobs
never touch this layer; persistent jobs are written to a small SQLite
database for the index and to a per-job directory for the artifacts.

The repository surface is intentionally small and free of SQLite-specific
constructs so that a future Postgres backend can implement
:class:`JobRepository` without changing any caller. The SQLite
implementation uses parameterised queries, ISO-8601 strings for
timestamps, and JSON for embedded payloads — all portable to Postgres.

Postgres-readiness checklist:

- ``JobRepository`` is a :class:`typing.Protocol`. A
  ``PostgresJobRepository`` subclass would only need to swap the
  ``?`` placeholders for ``%s`` and rewrite the ``CREATE TABLE``
  statement.
- Connections are owned by the repository; no caller imports
  ``sqlite3`` directly.
- All timestamps are persisted as ISO-8601 strings (UTC). Pydantic
  handles the round trip transparently.
- All booleans are persisted as integers ``0``/``1`` to keep the
  schema compatible with SQLite's storage classes.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from nextext.api.schemas import JobOptions, JobStatus


def resolve_data_dir() -> Path:
    """Resolve the on-disk root for persistent job storage.

    Order of resolution:

    1. ``NEXTEXT_DATA_DIR`` environment variable.
    2. ``./.nextext-data`` relative to the current working directory.

    The directory is created (with parents) if it does not exist.

    Returns:
        Path: The data root directory, guaranteed to exist.
    """
    raw = os.getenv("NEXTEXT_DATA_DIR")
    root = Path(raw) if raw else Path.cwd() / ".nextext-data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _utcnow() -> datetime:
    """Return a timezone-aware UTC ``datetime``.

    Returns:
        datetime: Current time with explicit UTC offset.
    """
    return datetime.now(tz=UTC)


def _iso(value: datetime | None) -> str | None:
    """Serialize a ``datetime`` to ISO-8601, or ``None``.

    Args:
        value: Timezone-aware datetime, or ``None``.

    Returns:
        str | None: ISO-8601 representation, or ``None``.
    """
    return value.isoformat() if value is not None else None


def _parse_iso(value: str | None) -> datetime | None:
    """Parse an ISO-8601 string back into a ``datetime``.

    Args:
        value: ISO-8601 string, or ``None``.

    Returns:
        datetime | None: Parsed datetime, or ``None``.
    """
    return datetime.fromisoformat(value) if value else None


@dataclass(slots=True)
class JobRecord:
    """Persistence-layer DTO that mirrors one row of the ``jobs`` table.

    This is intentionally separate from :class:`nextext.api.jobs.JobState`,
    which carries non-serializable runtime state (asyncio queues, the
    in-memory pipeline result dict, the matplotlib ``Figure``).
    """

    job_id: str
    owner_id: str
    status: JobStatus
    stage: str | None
    stage_index: int
    progress: float
    error: str | None
    file_name: str
    source_file_hash: str | None
    options: JobOptions
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    artifact_dir: str


@runtime_checkable
class JobRepository(Protocol):
    """Durable storage surface for opt-in persistent jobs.

    Implementations must be safe to call from any thread. The SQLite
    implementation serialises writes through a single connection guarded
    by a lock; a future Postgres implementation can rely on the server.
    """

    def init(self) -> None:
        """Create the schema if it does not yet exist."""

    def create(self, record: JobRecord) -> None:
        """Insert a new job row.

        Args:
            record: Job to persist.
        """

    def update_progress(
        self,
        job_id: str,
        *,
        stage: str | None,
        stage_index: int,
        progress: float,
    ) -> None:
        """Update the live progress fields on a row.

        Args:
            job_id: Job identifier.
            stage: Human-readable stage label.
            stage_index: Zero-based pipeline stage index.
            progress: Normalised progress in ``[0, 1]``.
        """

    def mark_running(self, job_id: str, started_at: datetime) -> None:
        """Mark a queued job as running.

        Args:
            job_id: Job identifier.
            started_at: Timestamp when the worker picked up the job.
        """

    def mark_completed(self, job_id: str, finished_at: datetime) -> None:
        """Mark a running job as completed.

        Args:
            job_id: Job identifier.
            finished_at: Timestamp when the worker finished the job.
        """

    def mark_failed(
        self,
        job_id: str,
        *,
        error: str,
        finished_at: datetime,
    ) -> None:
        """Mark a running job as failed.

        Args:
            job_id: Job identifier.
            error: Human-readable error message.
            finished_at: Timestamp when the worker recorded the failure.
        """

    def reset_running_to_interrupted(self) -> int:
        """Rewrite any ``running``/``queued`` row to ``interrupted``.

        Called once at backend startup so that jobs whose worker died
        with the container are visible to the user as failed rather than
        appearing to make progress forever.

        Returns:
            int: Number of rows updated.
        """

    def get(self, job_id: str) -> JobRecord | None:
        """Fetch a single row.

        Args:
            job_id: Job identifier.

        Returns:
            JobRecord | None: The row, or ``None`` if missing.
        """

    def list_for_owner(
        self,
        owner_id: str,
        statuses: list[JobStatus] | None = None,
    ) -> list[JobRecord]:
        """List rows owned by ``owner_id``, newest first.

        Args:
            owner_id: Cookie-derived owner identifier.
            statuses: Optional whitelist of statuses to include.

        Returns:
            list[JobRecord]: Matching rows ordered by ``created_at``
                descending.
        """

    def iter_all(self) -> Iterator[JobRecord]:
        """Iterate every persisted row in stable order.

        Yields:
            JobRecord: One row per emit.
        """

    def delete(self, job_id: str, owner_id: str) -> bool:
        """Delete a row when ``owner_id`` matches.

        Args:
            job_id: Job identifier.
            owner_id: Cookie-derived owner identifier.

        Returns:
            bool: ``True`` if a row was removed, ``False`` otherwise.
        """

    def close(self) -> None:
        """Close the underlying connection."""


class SqliteJobRepository:
    """SQLite-backed :class:`JobRepository` using stdlib ``sqlite3``.

    Suitable for single-container deployments. Multi-container setups
    should swap in a Postgres implementation; the public surface above
    is the only contract a replacement must satisfy.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        owner_id TEXT NOT NULL,
        status TEXT NOT NULL,
        stage TEXT,
        stage_index INTEGER NOT NULL DEFAULT 0,
        progress REAL NOT NULL DEFAULT 0.0,
        error TEXT,
        file_name TEXT NOT NULL,
        source_file_hash TEXT,
        options_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT,
        artifact_dir TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_jobs_owner_created
        ON jobs(owner_id, created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
    """

    def __init__(self, db_path: Path) -> None:
        """Open a connection rooted at ``db_path``.

        Args:
            db_path: Path to the SQLite file. Parent directory must
                already exist.
        """
        self._db_path = db_path
        # ``check_same_thread=False`` lets us share the connection across
        # the FastAPI request thread and the worker thread; ``_lock``
        # serialises writes so we never tear a transaction.
        self._conn = sqlite3.connect(
            db_path,
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False,
            isolation_level=None,  # autocommit; explicit BEGIN/COMMIT around writes
        )
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def init(self) -> None:
        """Create the schema if it does not yet exist."""
        with self._lock:
            self._conn.executescript(self._SCHEMA)

    def create(self, record: JobRecord) -> None:
        """Insert a new job row.

        Args:
            record: Job to persist.
        """
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO jobs (
                    job_id, owner_id, status, stage, stage_index,
                    progress, error, file_name, source_file_hash,
                    options_json, created_at, started_at, finished_at,
                    artifact_dir
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.job_id,
                    record.owner_id,
                    record.status.value,
                    record.stage,
                    record.stage_index,
                    record.progress,
                    record.error,
                    record.file_name,
                    record.source_file_hash,
                    record.options.model_dump_json(),
                    _iso(record.created_at),
                    _iso(record.started_at),
                    _iso(record.finished_at),
                    record.artifact_dir,
                ),
            )

    def update_progress(
        self,
        job_id: str,
        *,
        stage: str | None,
        stage_index: int,
        progress: float,
    ) -> None:
        """Update the live progress fields on a row.

        Args:
            job_id: Job identifier.
            stage: Human-readable stage label.
            stage_index: Zero-based pipeline stage index.
            progress: Normalised progress in ``[0, 1]``.
        """
        with self._lock:
            self._conn.execute(
                """
                UPDATE jobs
                   SET stage = ?, stage_index = ?, progress = ?
                 WHERE job_id = ?
                """,
                (stage, stage_index, progress, job_id),
            )

    def mark_running(self, job_id: str, started_at: datetime) -> None:
        """Mark a queued job as running.

        Args:
            job_id: Job identifier.
            started_at: Timestamp when the worker picked up the job.
        """
        with self._lock:
            self._conn.execute(
                """
                UPDATE jobs
                   SET status = ?, started_at = ?
                 WHERE job_id = ?
                """,
                (JobStatus.RUNNING.value, _iso(started_at), job_id),
            )

    def mark_completed(self, job_id: str, finished_at: datetime) -> None:
        """Mark a running job as completed.

        Args:
            job_id: Job identifier.
            finished_at: Timestamp when the worker finished the job.
        """
        with self._lock:
            self._conn.execute(
                """
                UPDATE jobs
                   SET status = ?, progress = 1.0, stage = NULL,
                       finished_at = ?
                 WHERE job_id = ?
                """,
                (JobStatus.COMPLETED.value, _iso(finished_at), job_id),
            )

    def mark_failed(
        self,
        job_id: str,
        *,
        error: str,
        finished_at: datetime,
    ) -> None:
        """Mark a running job as failed.

        Args:
            job_id: Job identifier.
            error: Human-readable error message.
            finished_at: Timestamp when the worker recorded the failure.
        """
        with self._lock:
            self._conn.execute(
                """
                UPDATE jobs
                   SET status = ?, error = ?, finished_at = ?
                 WHERE job_id = ?
                """,
                (JobStatus.FAILED.value, error, _iso(finished_at), job_id),
            )

    def reset_running_to_interrupted(self) -> int:
        """Rewrite any ``running``/``queued`` row to ``interrupted``.

        Returns:
            int: Number of rows updated.
        """
        finished_at = _iso(_utcnow())
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE jobs
                   SET status = 'interrupted',
                       error = COALESCE(error,
                                        'Backend restarted before this job finished.'),
                       finished_at = COALESCE(finished_at, ?)
                 WHERE status IN ('queued', 'running')
                """,
                (finished_at,),
            )
            return cursor.rowcount

    def get(self, job_id: str) -> JobRecord | None:
        """Fetch a single row.

        Args:
            job_id: Job identifier.

        Returns:
            JobRecord | None: The row, or ``None`` if missing.
        """
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return _row_to_record(row) if row else None

    def list_for_owner(
        self,
        owner_id: str,
        statuses: list[JobStatus] | None = None,
    ) -> list[JobRecord]:
        """List rows owned by ``owner_id``, newest first.

        Args:
            owner_id: Cookie-derived owner identifier.
            statuses: Optional whitelist of statuses to include.

        Returns:
            list[JobRecord]: Matching rows.
        """
        if statuses:
            placeholders = ",".join("?" * len(statuses))
            sql = f"SELECT * FROM jobs WHERE owner_id = ? AND status IN ({placeholders}) ORDER BY created_at DESC"
            params: tuple[Any, ...] = (owner_id, *(s.value for s in statuses))
        else:
            sql = "SELECT * FROM jobs WHERE owner_id = ? ORDER BY created_at DESC"
            params = (owner_id,)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_record(row) for row in rows]

    def iter_all(self) -> Iterator[JobRecord]:
        """Iterate every persisted row in stable order.

        Yields:
            JobRecord: One row per emit.
        """
        with self._lock:
            rows = self._conn.execute("SELECT * FROM jobs ORDER BY created_at ASC").fetchall()
        for row in rows:
            yield _row_to_record(row)

    def delete(self, job_id: str, owner_id: str) -> bool:
        """Delete a row when ``owner_id`` matches.

        Args:
            job_id: Job identifier.
            owner_id: Cookie-derived owner identifier.

        Returns:
            bool: ``True`` if a row was removed.
        """
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM jobs WHERE job_id = ? AND owner_id = ?",
                (job_id, owner_id),
            )
            return cursor.rowcount > 0

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            self._conn.close()


def _row_to_record(row: sqlite3.Row) -> JobRecord:
    """Materialise a ``sqlite3.Row`` into a ``JobRecord``.

    Args:
        row: Row returned by ``sqlite3`` with ``Row`` factory.

    Returns:
        JobRecord: Parsed record.
    """
    options = JobOptions.model_validate(json.loads(row["options_json"]))
    return JobRecord(
        job_id=row["job_id"],
        owner_id=row["owner_id"],
        status=JobStatus(row["status"]),
        stage=row["stage"],
        stage_index=row["stage_index"],
        progress=row["progress"],
        error=row["error"],
        file_name=row["file_name"],
        source_file_hash=row["source_file_hash"],
        options=options,
        created_at=_parse_iso(row["created_at"]),  # type: ignore[arg-type]
        started_at=_parse_iso(row["started_at"]),
        finished_at=_parse_iso(row["finished_at"]),
        artifact_dir=row["artifact_dir"],
    )


# ----------------------------------------------------------------------- files


@dataclass(slots=True)
class ArtifactStore:
    """Filesystem layout for one job's persisted artifacts.

    Each job lives under ``<data_root>/jobs/<job_id>/`` with one file per
    artifact type. Empty/absent files signal that the corresponding
    pipeline stage was skipped — this matches the in-memory result dict
    semantics, so :func:`render_artifact` can branch the same way.
    """

    job_id: str
    root: Path

    @classmethod
    def for_job(cls, data_root: Path, job_id: str) -> ArtifactStore:
        """Build the store for ``job_id`` under ``data_root``.

        Args:
            data_root: Persistence root (see :func:`resolve_data_dir`).
            job_id: Job identifier.

        Returns:
            ArtifactStore: New store. The directory is *not* created yet.
        """
        return cls(job_id=job_id, root=data_root / "jobs" / job_id)

    @property
    def relative(self) -> str:
        """Return the path of this store relative to ``jobs/``.

        Returns:
            str: A repository-friendly relative path.
        """
        return f"jobs/{self.job_id}"

    def ensure(self) -> None:
        """Create the per-job directory if it does not yet exist."""
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        """Return the on-disk path for one artifact name.

        Args:
            name: File name within the per-job directory
                (e.g. ``transcript.parquet``).

        Returns:
            Path: Absolute path inside the store.
        """
        return self.root / name

    def exists(self, name: str) -> bool:
        """Return whether the named artifact has been written.

        Args:
            name: File name within the per-job directory.

        Returns:
            bool: ``True`` if the file exists and is non-empty.
        """
        candidate = self.path(name)
        return candidate.is_file() and candidate.stat().st_size > 0

    def remove(self) -> None:
        """Delete the per-job directory and everything inside it."""
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)


def init_repository(db_path: Path | None = None) -> SqliteJobRepository:
    """Construct the default SQLite repository and create its schema.

    Args:
        db_path: Optional override; defaults to ``<data_root>/jobs.db``.

    Returns:
        SqliteJobRepository: Ready-to-use repository.
    """
    if db_path is None:
        db_path = resolve_data_dir() / "jobs.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    repo = SqliteJobRepository(db_path)
    repo.init()
    logger.info("Job persistence repository ready at {}", db_path)
    return repo


__all__ = [
    "ArtifactStore",
    "JobRecord",
    "JobRepository",
    "SqliteJobRepository",
    "init_repository",
    "resolve_data_dir",
]
