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
from typing import Any

import pandas as pd
import pytest

from nextext.api import artifacts
from nextext.api.jobs import JobState
from nextext.api.schemas import JobOptions, JobStatus


def _completed_job(name: str, *, summary: str | None = None) -> JobState:
    """Build a completed in-memory job carrying a small transcript result.

    Args:
        name: Upload file name; drives the archive folder and member stem.
        summary: Optional summary text; when given, populates ``result["summary"]``
            so the job contributes to the combined ``batch_summaries.txt``.

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
    result: dict[str, Any] = {"transcript": transcript, "resolved_src_lang": "en"}
    if summary is not None:
        result["summary"] = summary
    return JobState(
        job_id=f"job-{name}",
        owner_id="owner",
        file_name=name,
        file_path=Path(name),
        source_file_hash="sha256:deadbeef",
        options=options,
        status=JobStatus.COMPLETED,
        result=result,
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


def test_batch_archive_lists_every_summary_in_one_root_file() -> None:
    """A batch of summarized jobs gains a root ``batch_summaries.txt`` manifest.

    The combined file carries every file's name (with extension) and its full
    summary text, in the order the jobs are passed.
    """
    first = _completed_job("interview_2024.mp4", summary="Quarterly results and outlook.")
    second = _completed_job("keynote.mp3", summary="Product roadmap and milestones.")

    batch = zipfile.ZipFile(io.BytesIO(artifacts.build_batch_archive([first, second])))

    assert "batch_summaries.txt" in batch.namelist(), "combined summaries file missing"
    combined = batch.read("batch_summaries.txt").decode("utf-8")
    assert "interview_2024.mp4" in combined
    assert "Quarterly results and outlook." in combined
    assert "keynote.mp3" in combined
    assert "Product roadmap and milestones." in combined
    # Manifest order follows the job order.
    assert combined.index("interview_2024.mp4") < combined.index("keynote.mp3")


def test_batch_archive_omits_summaries_file_when_no_job_has_a_summary() -> None:
    """With no summaries in the batch, no ``batch_summaries.txt`` is added."""
    batch = zipfile.ZipFile(
        io.BytesIO(artifacts.build_batch_archive([_completed_job("a.wav"), _completed_job("b.wav")]))
    )

    assert "batch_summaries.txt" not in batch.namelist()


def test_batch_summaries_file_covers_only_summarized_jobs() -> None:
    """Only summary-bearing jobs appear in the combined file; others keep their folder."""
    summarized = _completed_job("with_summary.wav", summary="A real summary.")
    plain = _completed_job("no_summary.wav")

    batch = zipfile.ZipFile(io.BytesIO(artifacts.build_batch_archive([summarized, plain])))
    combined = batch.read("batch_summaries.txt").decode("utf-8")

    assert "with_summary.wav" in combined
    assert "no_summary.wav" not in combined
    # The summary-less job still contributes its own output folder.
    assert any(name.startswith("no_summary/") for name in batch.namelist())
