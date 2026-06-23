"""Unit tests for archive-member render caching in ``nextext.api.artifacts``.

These pin the performance contract behind the batch-download fix: the expensive
per-job render (XLSX serialization, etc.) is produced once and reused across
both the per-job archive and the combined batch archive, instead of each job's
archive being decompressed and re-compressed for the batch.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from nextext.api import artifacts
from nextext.api.jobs import JobState
from nextext.api.schemas import JobOptions, JobStatus


def _completed_job(name: str) -> JobState:
    """Build a completed in-memory job carrying a small transcript result.

    Args:
        name: Upload file name; drives the archive folder and member stem.

    Returns:
        JobState: A COMPLETED job whose result holds a two-row transcript.
    """
    options = JobOptions.model_validate(
        {
            "task": "transcribe",
            "trg_lang": "de",
            "speakers": 1,
            "words": False,
            "summarization": False,
            "hate_speech": False,
        }
    )
    transcript = pd.DataFrame(
        {
            "start": [0.0, 1.0],
            "end": [1.0, 2.0],
            "speaker": ["", ""],
            "text": ["hello", "world"],
        }
    )
    return JobState(
        job_id=f"job-{name}",
        owner_id="owner",
        file_name=name,
        file_path=Path(name),
        source_file_hash="sha256:deadbeef",
        options=options,
        status=JobStatus.COMPLETED,
        result={"transcript": transcript, "resolved_src_lang": "en"},
    )


def _by_basename(zf: zipfile.ZipFile) -> dict[str, bytes]:
    """Map each archive member to its decompressed bytes, keyed by base name.

    Args:
        zf: An open ZIP archive.

    Returns:
        dict[str, bytes]: ``filename -> bytes`` ignoring the leading folder.
    """
    return {Path(name).name: zf.read(name) for name in zf.namelist()}


def test_render_runs_once_across_per_job_and_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The costly XLSX render runs once per job across both archive paths.

    Building the per-job archive and then the batch archive for the same job
    must not render the transcript XLSX twice; the cached members are reused.

    Args:
        monkeypatch: Pytest fixture used to count ``_df_to_xlsx`` invocations.
    """
    calls = 0
    real_to_xlsx = artifacts._df_to_xlsx

    def _counting(df: pd.DataFrame) -> bytes:
        nonlocal calls
        calls += 1
        return real_to_xlsx(df)

    monkeypatch.setattr(artifacts, "_df_to_xlsx", _counting)

    job = _completed_job("clip.wav")
    artifacts.build_archive_for_job(job)
    artifacts.build_batch_archive([job])

    assert calls == 1, "transcript XLSX should be rendered once and reused, not per archive"


def test_batch_member_bytes_match_per_job() -> None:
    """A single-job batch archive holds the same file bytes as its per-job archive."""
    job = _completed_job("clip.wav")
    per_job = zipfile.ZipFile(io.BytesIO(artifacts.build_archive_for_job(job)))
    batch = zipfile.ZipFile(io.BytesIO(artifacts.build_batch_archive([job])))

    assert _by_basename(per_job) == _by_basename(batch)
