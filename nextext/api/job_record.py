"""The owner record kept next to each job's upload.

Jobs and their results live in memory, but their uploads live on disk, in the
job directory on the ``tmp-jobs`` volume, so a backend that dies without
shutting down cleanly leaves uploads behind that no live job owns. Each job
directory therefore also holds a small record of the job's id and owner
(``job.json``), written atomically when the job is created. The sweep that
removes such leftovers at the next startup reads it to record whose data it
deleted — the attribution a restart would otherwise lose.

A directory without a valid record — an upload interrupted before its job
existed, or one left by a release that wrote no records — is never served
(nothing in memory points at it) and is removed by the same sweep; its
deletion is recorded with neither a job id nor an owner.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = ["RECORD_NAME", "JobRecord", "read_job_record", "write_job_record"]

#: File name of the record inside a job directory.
RECORD_NAME: Final = "job.json"
_RECORD_VERSION: Final = 1
_MAX_RECORD_BYTES: Final = 4096
_JOB_ID: Final = re.compile(r"[0-9a-f]{32}")
_CONTROL: Final = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")


@dataclass(frozen=True)
class JobRecord:
    """The durable facts about a job: which job, whose, since when.

    Attributes:
        job_id: The job id (32 lowercase hex digits).
        owner_id: The principal that owns the job.
        created_at: When the job was created, ISO 8601.
    """

    job_id: str
    owner_id: str
    created_at: str


def write_job_record(job_dir: Path, record: JobRecord) -> None:
    """Write a job's record into its directory, atomically.

    The record is written to a temporary file, flushed to disk, and renamed
    over ``job.json``, so a reader sees either no record or a whole one.

    Args:
        job_dir: The job's directory.
        record: The record to write.

    Raises:
        OSError: If the record cannot be written; the job must not start then,
            since a crash would leave its upload unattributed.
    """
    payload = json.dumps(
        {
            "v": _RECORD_VERSION,
            "job_id": record.job_id,
            "owner_id": record.owner_id,
            "created_at": record.created_at,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    tmp_path = job_dir / f".{RECORD_NAME}.tmp"
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC, 0o600)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(fd, view) :]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp_path, job_dir / RECORD_NAME)


def read_job_record(job_dir: Path) -> JobRecord | None:
    """Read a job's record, if it has a valid one.

    Args:
        job_dir: The job's directory.

    Returns:
        JobRecord | None: The record, or ``None`` when it is missing,
            unreadable, oversized, or malformed.
    """
    path = job_dir / RECORD_NAME
    try:
        if path.is_symlink() or path.stat().st_size > _MAX_RECORD_BYTES:
            return None
        data = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("v") != _RECORD_VERSION:
        return None
    job_id, owner_id, created_at = data.get("job_id"), data.get("owner_id"), data.get("created_at")
    if not (isinstance(job_id, str) and _JOB_ID.fullmatch(job_id)):
        return None
    if not (isinstance(owner_id, str) and owner_id.strip()):
        return None
    if _CONTROL.search(owner_id) or not isinstance(created_at, str):
        return None
    return JobRecord(job_id=job_id, owner_id=owner_id, created_at=created_at)
