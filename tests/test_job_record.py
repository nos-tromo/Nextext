"""Tests for the owner record kept next to each job's upload (``job.json``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nextext.api.job_record import RECORD_NAME, JobRecord, read_job_record, write_job_record

RECORD = JobRecord(job_id="a1" * 16, owner_id="alice.example", created_at="2026-01-31T12:00:00+00:00")


def test_record_round_trips(tmp_path: Path) -> None:
    """What is written is read back unchanged, and no temp file is left over."""
    write_job_record(tmp_path, RECORD)
    assert read_job_record(tmp_path) == RECORD
    assert sorted(path.name for path in tmp_path.iterdir()) == [RECORD_NAME]
    assert (tmp_path / RECORD_NAME).stat().st_mode & 0o077 == 0


def test_rewrite_replaces_the_record_whole(tmp_path: Path) -> None:
    """A second write replaces the first in one rename."""
    write_job_record(tmp_path, RECORD)
    other = JobRecord(job_id="b2" * 16, owner_id="service:docint", created_at="2026-02-01T08:00:00+00:00")
    write_job_record(tmp_path, other)
    assert read_job_record(tmp_path) == other


def test_missing_record_reads_as_none(tmp_path: Path) -> None:
    """A directory without a record has no recorded owner."""
    assert read_job_record(tmp_path) is None


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"{not json", id="not-json"),
        pytest.param(b"\xff\xfe", id="not-utf8"),
        pytest.param(json.dumps([1, 2]).encode(), id="not-an-object"),
        pytest.param(json.dumps({"v": 2, "job_id": "a1" * 16, "owner_id": "o", "created_at": "t"}).encode(), id="v2"),
        pytest.param(json.dumps({"v": 1, "job_id": "short", "owner_id": "o", "created_at": "t"}).encode(), id="job-id"),
        pytest.param(
            json.dumps({"v": 1, "job_id": "a1" * 16, "owner_id": "  ", "created_at": "t"}).encode(), id="blank"
        ),
        pytest.param(
            json.dumps({"v": 1, "job_id": "a1" * 16, "owner_id": "a\nb", "created_at": "t"}).encode(), id="control"
        ),
        pytest.param(json.dumps({"v": 1, "job_id": "a1" * 16, "owner_id": "o", "created_at": 5}).encode(), id="ts"),
        pytest.param(b" " * 5000, id="oversized"),
    ],
)
def test_malformed_record_reads_as_none(tmp_path: Path, content: bytes) -> None:
    """Anything but a whole, valid record counts as no record.

    Args:
        tmp_path: The job directory.
        content: The record file's bytes.
    """
    (tmp_path / RECORD_NAME).write_bytes(content)
    assert read_job_record(tmp_path) is None


def test_symlinked_record_is_not_followed(tmp_path: Path) -> None:
    """A record that is a symlink is ignored, wherever it points."""
    elsewhere = tmp_path / "elsewhere.json"
    write_job_record(tmp_path, RECORD)
    (tmp_path / RECORD_NAME).rename(elsewhere)
    (tmp_path / RECORD_NAME).symlink_to(elsewhere)
    assert read_job_record(tmp_path) is None
