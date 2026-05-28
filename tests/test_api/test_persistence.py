"""Tests for the SQLite-backed job repository and filesystem helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from nextext.api.persistence import (
    ArtifactStore,
    JobRecord,
    SqliteJobRepository,
    init_repository,
    resolve_data_dir,
)
from nextext.api.schemas import JobOptions, JobStatus


def _make_record(
    *,
    job_id: str = "job-1",
    owner_id: str = "owner-1",
    status: JobStatus = JobStatus.QUEUED,
    file_name: str = "clip.wav",
) -> JobRecord:
    """Build a minimal ``JobRecord`` for tests.

    Args:
        job_id: Identifier for the row.
        owner_id: Owner identifier (cookie-derived).
        status: Initial lifecycle status.
        file_name: Original upload file name.

    Returns:
        JobRecord: Populated record with deterministic timestamps.
    """
    return JobRecord(
        job_id=job_id,
        owner_id=owner_id,
        status=status,
        stage=None,
        stage_index=0,
        progress=0.0,
        error=None,
        file_name=file_name,
        source_file_hash="sha256:deadbeef",
        options=JobOptions(persist=True),
        created_at=datetime(2026, 5, 1, tzinfo=UTC),
        started_at=None,
        finished_at=None,
        artifact_dir=f"jobs/{job_id}",
    )


@pytest.fixture
def repo(tmp_path: Path) -> SqliteJobRepository:
    """Provide a fresh SQLite repository rooted in ``tmp_path``.

    Yields:
        SqliteJobRepository: A repository whose connection is closed at
            the end of the test.
    """
    repository = SqliteJobRepository(tmp_path / "jobs.db")
    repository.init()
    try:
        return repository
    finally:
        pass  # closed in test teardown via the autouse cleanup below


@pytest.fixture(autouse=True)
def _close_repo(request: pytest.FixtureRequest) -> None:
    """Close any ``SqliteJobRepository`` fixture used by the test."""

    def _cleanup() -> None:
        repo = request.node.funcargs.get("repo")
        if isinstance(repo, SqliteJobRepository):
            repo.close()

    request.addfinalizer(_cleanup)


def test_resolve_data_dir_honours_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``NEXTEXT_DATA_DIR`` should override the default location."""
    target = tmp_path / "custom"
    monkeypatch.setenv("NEXTEXT_DATA_DIR", str(target))
    resolved = resolve_data_dir()
    assert resolved == target
    assert resolved.is_dir()


def test_create_and_get_round_trip(repo: SqliteJobRepository) -> None:
    """Inserting a record and fetching it back preserves every field."""
    record = _make_record()
    repo.create(record)
    fetched = repo.get(record.job_id)
    assert fetched is not None
    assert fetched.job_id == record.job_id
    assert fetched.owner_id == record.owner_id
    assert fetched.options.persist is True
    assert fetched.created_at == record.created_at


def test_update_progress_persists_stage_index_and_progress(
    repo: SqliteJobRepository,
) -> None:
    """``update_progress`` should update the live fields without touching status."""
    record = _make_record()
    repo.create(record)
    repo.update_progress(
        record.job_id,
        stage="Transcribing",
        stage_index=1,
        progress=0.25,
    )
    fetched = repo.get(record.job_id)
    assert fetched is not None
    assert fetched.stage == "Transcribing"
    assert fetched.stage_index == 1
    assert fetched.progress == pytest.approx(0.25)
    assert fetched.status == JobStatus.QUEUED


def test_mark_completed_sets_terminal_state(repo: SqliteJobRepository) -> None:
    """``mark_completed`` should set status, finished_at, and progress=1."""
    record = _make_record()
    repo.create(record)
    finished = datetime(2026, 5, 1, 1, 0, tzinfo=UTC)
    repo.mark_completed(record.job_id, finished_at=finished)
    fetched = repo.get(record.job_id)
    assert fetched is not None
    assert fetched.status == JobStatus.COMPLETED
    assert fetched.finished_at == finished
    assert fetched.progress == pytest.approx(1.0)


def test_mark_failed_records_error_message(repo: SqliteJobRepository) -> None:
    """``mark_failed`` should preserve the operator-facing error."""
    record = _make_record()
    repo.create(record)
    finished = datetime(2026, 5, 1, 0, 5, tzinfo=UTC)
    repo.mark_failed(record.job_id, error="boom", finished_at=finished)
    fetched = repo.get(record.job_id)
    assert fetched is not None
    assert fetched.status == JobStatus.FAILED
    assert fetched.error == "boom"
    assert fetched.finished_at == finished


def test_reset_running_to_interrupted_rewrites_unfinished_rows(
    repo: SqliteJobRepository,
) -> None:
    """Container restart should leave no rows in ``running``/``queued``."""
    repo.create(_make_record(job_id="job-a", status=JobStatus.QUEUED))
    repo.create(_make_record(job_id="job-b", status=JobStatus.RUNNING))
    repo.create(_make_record(job_id="job-c", status=JobStatus.COMPLETED))

    updated = repo.reset_running_to_interrupted()

    assert updated == 2
    fetched_c = repo.get("job-c")
    assert fetched_c is not None
    assert fetched_c.status == JobStatus.COMPLETED
    for job_id in ("job-a", "job-b"):
        fetched = repo.get(job_id)
        assert fetched is not None
        assert fetched.status == JobStatus.INTERRUPTED
        assert fetched.error is not None
        assert "restart" in fetched.error.lower()


def test_list_for_owner_is_scoped_and_ordered(repo: SqliteJobRepository) -> None:
    """The listing endpoint should only see one owner's rows, newest first."""
    repo.create(_make_record(job_id="alice-1", owner_id="alice"))
    repo.create(
        JobRecord(
            job_id="alice-2",
            owner_id="alice",
            status=JobStatus.COMPLETED,
            stage=None,
            stage_index=0,
            progress=1.0,
            error=None,
            file_name="b.wav",
            source_file_hash=None,
            options=JobOptions(persist=True),
            created_at=datetime(2026, 5, 2, tzinfo=UTC),
            started_at=None,
            finished_at=None,
            artifact_dir="jobs/alice-2",
        )
    )
    repo.create(_make_record(job_id="bob-1", owner_id="bob"))

    alice_jobs = repo.list_for_owner("alice")
    assert [j.job_id for j in alice_jobs] == ["alice-2", "alice-1"]
    assert all(j.owner_id == "alice" for j in alice_jobs)
    assert repo.list_for_owner("bob")[0].job_id == "bob-1"


def test_list_for_owner_filters_by_status(repo: SqliteJobRepository) -> None:
    """Optional status whitelist should constrain results."""
    repo.create(_make_record(job_id="q-1", status=JobStatus.QUEUED))
    repo.create(_make_record(job_id="c-1", status=JobStatus.COMPLETED))
    listed = repo.list_for_owner("owner-1", statuses=[JobStatus.COMPLETED])
    assert [job.job_id for job in listed] == ["c-1"]


def test_delete_enforces_owner_check(repo: SqliteJobRepository) -> None:
    """Wrong-owner deletes must report failure without touching the row."""
    repo.create(_make_record(job_id="job-x", owner_id="alice"))
    assert repo.delete("job-x", owner_id="bob") is False
    assert repo.get("job-x") is not None
    assert repo.delete("job-x", owner_id="alice") is True
    assert repo.get("job-x") is None


def test_iter_all_yields_every_row(repo: SqliteJobRepository) -> None:
    """``iter_all`` powers the startup rehydration loop."""
    repo.create(_make_record(job_id="job-1"))
    repo.create(_make_record(job_id="job-2"))
    rows = list(repo.iter_all())
    assert {row.job_id for row in rows} == {"job-1", "job-2"}


def test_init_repository_creates_db_and_schema(tmp_path: Path) -> None:
    """The convenience constructor should produce a usable repository."""
    db_path = tmp_path / "data" / "jobs.db"
    repo = init_repository(db_path=db_path)
    try:
        assert db_path.is_file()
        repo.create(_make_record())
        assert repo.get("job-1") is not None
    finally:
        repo.close()


def test_artifact_store_lifecycle(tmp_path: Path) -> None:
    """The artifact store should create, expose, and remove its directory."""
    store = ArtifactStore.for_job(tmp_path, "job-7")
    assert store.relative == "jobs/job-7"
    assert not store.exists("transcript.parquet")
    store.ensure()
    target = store.path("transcript.parquet")
    target.write_bytes(b"hi")
    assert store.exists("transcript.parquet")
    store.remove()
    assert not store.root.exists()


def test_artifact_store_treats_empty_file_as_missing(tmp_path: Path) -> None:
    """Zero-byte artifacts should report as missing to keep the 404 path honest."""
    store = ArtifactStore.for_job(tmp_path, "job-9")
    store.ensure()
    store.path("summary.txt").write_bytes(b"")
    assert store.exists("summary.txt") is False
