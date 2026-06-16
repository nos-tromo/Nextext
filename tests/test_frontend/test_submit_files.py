"""Tests for the frontend batch submission helper.

These pin the memory-safety contract of the upload path: the frontend must
hand each Streamlit ``UploadedFile`` straight to the backend client so httpx
can stream it, instead of slurping the whole file into a second ``bytes``
copy before sending.
"""

from __future__ import annotations

import io
from typing import Any, cast

from nextext.frontend.app import _submit_files
from nextext.frontend.client import BackendClient


class _RecordingUpload(io.BytesIO):
    """A fake Streamlit ``UploadedFile`` that records whole-file reads."""

    def __init__(self, name: str, data: bytes) -> None:
        """Store the payload and expose the ``name``/``size`` UI attributes.

        Args:
            name: File name reported to the uploader.
            data: Raw file bytes.
        """
        super().__init__(data)
        self.name = name
        self.size = len(data)
        self.whole_reads = 0

    def read(self, size: int | None = -1) -> bytes:
        """Read bytes, counting reads that pull the entire remaining file.

        Args:
            size: Number of bytes to read; ``-1``/``None`` means read all.

        Returns:
            bytes: The bytes read.
        """
        if size is None or size < 0:
            self.whole_reads += 1
        return super().read(-1 if size is None else size)


class _RecordingClient:
    """A fake backend client capturing what ``submit_job`` receives."""

    def __init__(self) -> None:
        """Initialize the captured-call log."""
        self.calls: list[tuple[str, Any]] = []

    def submit_job(self, file_name: str, content: Any, options: dict[str, Any]) -> str:
        """Record the submission and return a synthetic job id.

        Args:
            file_name: Display name of the file.
            content: File payload handed to the client.
            options: Pipeline options dict.

        Returns:
            str: A synthetic job id.
        """
        self.calls.append((file_name, content))
        return f"job-{len(self.calls)}"


def test_submit_files_streams_file_object_without_whole_read() -> None:
    """``_submit_files`` passes the file object through, never reading it whole."""
    upload = _RecordingUpload("clip.wav", b"x" * 4096)
    client = _RecordingClient()

    jobs = _submit_files(cast(BackendClient, client), [upload], {"task": "transcribe"})

    assert jobs == [{"job_id": "job-1", "file_name": "clip.wav"}]
    # The raw file object reaches the client (so httpx can stream it in
    # chunks); the frontend must not materialise a second full bytes copy.
    assert client.calls[0][1] is upload
    assert upload.whole_reads == 0


def test_submit_files_returns_handles_for_every_file_in_order() -> None:
    """Every accepted file yields a job handle, preserving upload order."""
    uploads = [
        _RecordingUpload("a.wav", b"a"),
        _RecordingUpload("b.wav", b"bb"),
    ]
    client = _RecordingClient()

    jobs = _submit_files(cast(BackendClient, client), uploads, {"task": "transcribe"})

    assert jobs == [
        {"job_id": "job-1", "file_name": "a.wav"},
        {"job_id": "job-2", "file_name": "b.wav"},
    ]
