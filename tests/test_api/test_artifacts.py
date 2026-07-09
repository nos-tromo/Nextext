"""Tests for ``/api/v1/jobs/{id}/artifacts/{name}``."""

from __future__ import annotations

import io
import json
import time
import zipfile
from typing import Any, cast

import pandas as pd
from fastapi.testclient import TestClient

from nextext.api.jobs import JobManager


def _submit_and_wait(client: TestClient, options: dict[str, Any]) -> str:
    """Submit a job and block until it completes.

    Args:
        client: TestClient bound to the stub-app fixture.
        options: Pipeline options dictionary.

    Returns:
        str: The completed job's id.
    """
    response = client.post(
        "/api/v1/jobs",
        files={"file": ("clip.wav", io.BytesIO(b"x"), "audio/wav")},
        data={"options": json.dumps(options)},
    )
    assert response.status_code == 201
    job_id = cast(str, response.json()["job_id"])
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        snapshot = client.get(f"/api/v1/jobs/{job_id}").json()
        if snapshot["status"] == "completed":
            return job_id
        time.sleep(0.05)
    raise AssertionError(f"Job {job_id} did not complete in time.")


def test_transcript_csv_artifact_round_trips(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """Transcript CSV should parse back into the original two rows."""
    client, _ = stub_app_client
    job_id = _submit_and_wait(
        client,
        {
            "task": "transcribe",
            "trg_lang": "de",
            "speakers": 1,
            "words": False,
            "summarization": False,
            "hate_speech": False,
        },
    )
    response = client.get(f"/api/v1/jobs/{job_id}/artifacts/transcript.csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    df = pd.read_csv(io.BytesIO(response.content))
    assert list(df.columns) == ["start", "end", "speaker", "text"]
    assert len(df) == 2


def test_transcript_csv_artifact_includes_translation_column_for_translate_task(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """A translate-task job's transcript CSV should carry both text and translation columns."""
    client, _ = stub_app_client
    job_id = _submit_and_wait(
        client,
        {
            "task": "translate",
            "trg_lang": "de",
            "speakers": 1,
            "words": False,
            "summarization": False,
            "hate_speech": False,
        },
    )
    response = client.get(f"/api/v1/jobs/{job_id}/artifacts/transcript.csv")
    assert response.status_code == 200
    df = pd.read_csv(io.BytesIO(response.content))
    assert list(df.columns) == ["start", "end", "speaker", "text", "translation"]
    assert list(df["text"]) == ["Hello world.", "Second segment."]
    assert list(df["translation"]) == ["Hallo Welt.", "Zweites Segment."]


def test_docint_jsonl_includes_language(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """Each docint JSONL record should carry the resolved source language."""
    client, _ = stub_app_client
    job_id = _submit_and_wait(
        client,
        {
            "task": "transcribe",
            "trg_lang": "de",
            "speakers": 1,
            "words": False,
            "summarization": False,
            "hate_speech": False,
        },
    )
    response = client.get(f"/api/v1/jobs/{job_id}/artifacts/docint.jsonl")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(line) for line in response.content.decode("utf-8").splitlines() if line]
    assert lines
    # The stub pipeline resolves the source language to "en".
    for record in lines:
        assert record["language"] == "en"


def test_docint_jsonl_uses_original_text_for_translate_task(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """The docint JSONL's ``text`` field should carry the original transcript, not the translation.

    docint has its own ad-hoc translation for operators unfamiliar with the
    source language, so Nextext feeds it the highest-fidelity (untranslated)
    transcript regardless of task. ``language`` is therefore always the
    resolved source language, even though the job's ``transcript_language``
    (used for downstream analysis) is the target language for a translate
    task.
    """
    client, _ = stub_app_client
    job_id = _submit_and_wait(
        client,
        {
            "task": "translate",
            "trg_lang": "de",
            "speakers": 1,
            "words": False,
            "summarization": False,
            "hate_speech": False,
        },
    )
    response = client.get(f"/api/v1/jobs/{job_id}/artifacts/docint.jsonl")
    assert response.status_code == 200
    lines = [json.loads(line) for line in response.content.decode("utf-8").splitlines() if line]
    assert [record["text"] for record in lines] == ["Hello world.", "Second segment."]
    for record in lines:
        assert record["language"] == "en"


def test_summary_artifact_returns_404_when_not_requested(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """Missing artifacts should produce a 404, not an empty body."""
    client, _ = stub_app_client
    job_id = _submit_and_wait(
        client,
        {
            "task": "transcribe",
            "trg_lang": "de",
            "speakers": 1,
            "words": False,
            "summarization": False,
            "hate_speech": False,
        },
    )
    response = client.get(f"/api/v1/jobs/{job_id}/artifacts/summary.txt")
    assert response.status_code == 404


def test_archive_zip_contains_transcript_and_summary(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """The ZIP archive should bundle every produced output."""
    client, _ = stub_app_client
    job_id = _submit_and_wait(
        client,
        {
            "task": "transcribe",
            "trg_lang": "de",
            "speakers": 1,
            "words": False,
            "summarization": True,
            "hate_speech": False,
        },
    )
    response = client.get(f"/api/v1/jobs/{job_id}/artifacts/archive.zip")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = set(archive.namelist())
    assert any(name.endswith("_transcript.csv") for name in names)
    assert any(name.endswith("_transcript.xlsx") for name in names)
    assert any(name.endswith("_summary.txt") for name in names)


def test_unknown_artifact_returns_404(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """Unsupported artifact names should yield a 404."""
    client, _ = stub_app_client
    job_id = _submit_and_wait(
        client,
        {
            "task": "transcribe",
            "trg_lang": "de",
            "speakers": 1,
            "words": False,
            "summarization": False,
            "hate_speech": False,
        },
    )
    response = client.get(f"/api/v1/jobs/{job_id}/artifacts/secrets.tar")
    assert response.status_code == 404


def test_transcript_txt_artifact_is_tab_delimited(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """The transcript TXT artifact is a tab-delimited table with a header row."""
    client, _ = stub_app_client
    job_id = _submit_and_wait(
        client,
        {
            "task": "transcribe",
            "trg_lang": "de",
            "speakers": 1,
            "words": False,
            "summarization": False,
            "hate_speech": False,
        },
    )
    response = client.get(f"/api/v1/jobs/{job_id}/artifacts/transcript.txt")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.content.decode("utf-8").splitlines()[0] == "start\tend\tspeaker\ttext"
    df = pd.read_csv(io.BytesIO(response.content), sep="\t")
    assert list(df.columns) == ["start", "end", "speaker", "text"]
    assert len(df) == 2


def test_translation_txt_artifact_404_for_transcribe_task(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """translation.txt is absent (404) when the job did not translate."""
    client, _ = stub_app_client
    job_id = _submit_and_wait(
        client,
        {
            "task": "transcribe",
            "trg_lang": "de",
            "speakers": 1,
            "words": False,
            "summarization": False,
            "hate_speech": False,
        },
    )
    response = client.get(f"/api/v1/jobs/{job_id}/artifacts/translation.txt")
    assert response.status_code == 404


def test_translate_task_splits_transcript_and_translation_txt(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """A translate job exposes transcript.txt (source) and translation.txt (target) separately."""
    client, _ = stub_app_client
    job_id = _submit_and_wait(
        client,
        {
            "task": "translate",
            "trg_lang": "de",
            "speakers": 1,
            "words": False,
            "summarization": False,
            "hate_speech": False,
        },
    )
    transcript = client.get(f"/api/v1/jobs/{job_id}/artifacts/transcript.txt")
    assert transcript.status_code == 200
    transcript_df = pd.read_csv(io.BytesIO(transcript.content), sep="\t")
    assert list(transcript_df.columns) == ["start", "end", "speaker", "text"]
    assert list(transcript_df["text"]) == ["Hello world.", "Second segment."]

    translation = client.get(f"/api/v1/jobs/{job_id}/artifacts/translation.txt")
    assert translation.status_code == 200
    translation_df = pd.read_csv(io.BytesIO(translation.content), sep="\t")
    assert list(translation_df.columns) == ["start", "end", "speaker", "translation"]
    assert list(translation_df["translation"]) == ["Hallo Welt.", "Zweites Segment."]


def test_archive_zip_contains_txt_exports(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """The per-job ZIP bundles both split TXT files for a translate task."""
    client, _ = stub_app_client
    job_id = _submit_and_wait(
        client,
        {
            "task": "translate",
            "trg_lang": "de",
            "speakers": 1,
            "words": False,
            "summarization": False,
            "hate_speech": False,
        },
    )
    response = client.get(f"/api/v1/jobs/{job_id}/artifacts/archive.zip")
    assert response.status_code == 200
    names = set(zipfile.ZipFile(io.BytesIO(response.content)).namelist())
    assert any(name.endswith("_transcript.txt") for name in names)
    assert any(name.endswith("_translation.txt") for name in names)
