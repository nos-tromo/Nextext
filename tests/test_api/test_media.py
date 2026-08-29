"""Tests for ``/api/v1/jobs/{id}/media`` — the capability-URL media stream.

A ``<video src>`` cannot send the trusted identity header, so this route is
authorized by a per-job token carried in the URL rather than by the request
principal. These tests pin that contract, plus the HTTP Range support a
seek-to-timestamp player depends on.
"""

from __future__ import annotations

import io
import json
import time
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient

from nextext.api.jobs import JobManager, _normalize_transcript_row

MEDIA_BYTES = b"0123456789abcdef"


def _submit_and_wait(
    client: TestClient,
    *,
    file_name: str = "clip.wav",
    content: bytes = MEDIA_BYTES,
    content_type: str = "audio/wav",
) -> str:
    """Submit a job and block until it completes.

    Args:
        client: TestClient bound to the stub-app fixture.
        file_name: Upload filename, which drives the served MIME type.
        content: Upload payload bytes.
        content_type: Multipart content type of the upload part.

    Returns:
        str: The completed job's id.
    """
    options: dict[str, Any] = {"task": "transcribe", "trg_lang": "de"}
    response = client.post(
        "/api/v1/jobs",
        files={"file": (file_name, io.BytesIO(content), content_type)},
        data={"options": json.dumps(options)},
    )
    assert response.status_code == 201
    job_id = cast(str, response.json()["job_id"])
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed":
            return job_id
        time.sleep(0.05)
    raise AssertionError(f"Job {job_id} did not complete in time.")


def _media_url(client: TestClient, job_id: str) -> str:
    """Return the job's advertised media URL.

    Args:
        client: TestClient bound to the stub-app fixture.
        job_id: The completed job's id.

    Returns:
        str: The ``media_url`` from the job snapshot's result.
    """
    result = client.get(f"/api/v1/jobs/{job_id}").json()["result"]
    url = result["media_url"]
    assert isinstance(url, str)
    return url


# ---------------------------------------------------------------------------
# media_url on the snapshot
# ---------------------------------------------------------------------------


def test_completed_job_advertises_a_media_url(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """The result carries a media URL pointing at this job, with a token."""
    client, _ = stub_app_client
    job_id = _submit_and_wait(client)

    url = _media_url(client, job_id)

    parsed = urlparse(url)
    assert parsed.path == f"/api/v1/jobs/{job_id}/media"
    token = parse_qs(parsed.query).get("token", [""])[0]
    assert len(token) >= 20  # a real secret, not a placeholder


def test_media_token_is_unique_per_job(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """Two jobs get different tokens, so one URL never unlocks another job."""
    client, _ = stub_app_client
    first = _media_url(client, _submit_and_wait(client))
    second = _media_url(client, _submit_and_wait(client))
    assert first != second


def test_job_listing_never_exposes_the_media_token(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """The compact listing must not leak the capability to bulk readers."""
    client, _ = stub_app_client
    _submit_and_wait(client)
    assert "token" not in client.get("/api/v1/jobs").text


def test_media_url_absent_once_the_upload_is_gone(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """A job whose bytes were removed advertises no URL rather than a 404 one."""
    client, manager = stub_app_client
    job_id = _submit_and_wait(client)
    state = manager._jobs[job_id]
    state.file_path.unlink()

    assert client.get(f"/api/v1/jobs/{job_id}").json()["result"]["media_url"] is None


# ---------------------------------------------------------------------------
# Serving the bytes
# ---------------------------------------------------------------------------


def test_media_route_serves_the_original_bytes(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """The upload comes back verbatim."""
    client, _ = stub_app_client
    response = client.get(_media_url(client, _submit_and_wait(client)))
    assert response.status_code == 200
    assert response.content == MEDIA_BYTES


def test_media_route_needs_no_identity_header(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """A media element sends no custom headers, so the token must suffice.

    This is the whole reason the route exists in this shape: without it, a
    header-less ``<video>`` request would 401 or resolve to a different
    principal and 404.
    """
    client, _ = stub_app_client
    url = _media_url(client, _submit_and_wait(client))
    response = client.get(url, headers={"X-Auth-User": ""})
    assert response.status_code == 200


def test_media_route_guesses_the_content_type_from_the_filename(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """The browser picks <video> vs <audio> from this, so it must be right."""
    client, _ = stub_app_client
    url = _media_url(client, _submit_and_wait(client, file_name="clip.mp4"))
    assert client.get(url).headers["content-type"].startswith("video/mp4")


def test_media_route_serves_matroska_as_video(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """`.mkv` is in the upload allowlist but missing from stdlib mimetypes."""
    client, _ = stub_app_client
    url = _media_url(client, _submit_and_wait(client, file_name="clip.mkv"))
    assert client.get(url).headers["content-type"].startswith("video/")


def test_media_route_is_not_a_download(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """An `attachment` disposition would make the browser save, not play."""
    client, _ = stub_app_client
    response = client.get(_media_url(client, _submit_and_wait(client)))
    assert "attachment" not in response.headers.get("content-disposition", "")


# ---------------------------------------------------------------------------
# Range — what makes seeking cheap
# ---------------------------------------------------------------------------


def test_media_route_supports_range_requests(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """Seeking must fetch a slice, not re-download the whole recording."""
    client, _ = stub_app_client
    url = _media_url(client, _submit_and_wait(client))

    response = client.get(url, headers={"Range": "bytes=4-7"})

    assert response.status_code == 206
    assert response.content == MEDIA_BYTES[4:8]
    assert response.headers["content-range"] == f"bytes 4-7/{len(MEDIA_BYTES)}"


def test_media_route_advertises_range_support(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """Without `Accept-Ranges` the browser will not offer scrubbing at all."""
    client, _ = stub_app_client
    response = client.get(_media_url(client, _submit_and_wait(client)))
    assert response.headers.get("accept-ranges") == "bytes"


# ---------------------------------------------------------------------------
# Authorization failures — all 404, never 403
# ---------------------------------------------------------------------------


def test_media_route_rejects_a_wrong_token(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """A guessed token gets the same answer as a job that does not exist."""
    client, _ = stub_app_client
    job_id = _submit_and_wait(client)
    assert client.get(f"/api/v1/jobs/{job_id}/media?token=wrong").status_code == 404


def test_media_route_requires_a_token(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """No token is not the same as being the owner — the URL is the capability."""
    client, _ = stub_app_client
    job_id = _submit_and_wait(client)
    assert client.get(f"/api/v1/jobs/{job_id}/media").status_code in {404, 422}


def test_media_route_404s_for_an_unknown_job(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """An unknown id leaks nothing about whether it ever existed."""
    client, _ = stub_app_client
    assert client.get("/api/v1/jobs/deadbeef/media?token=whatever").status_code == 404


def test_media_route_404s_after_the_job_is_deleted(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """Deleting a job revokes its media URL along with its bytes."""
    client, _ = stub_app_client
    job_id = _submit_and_wait(client)
    url = _media_url(client, job_id)
    assert client.delete(f"/api/v1/jobs/{job_id}").status_code == 204

    assert client.get(url).status_code == 404


def test_media_route_404s_when_the_file_is_missing(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """A valid token whose bytes vanished must not raise a 500."""
    client, manager = stub_app_client
    job_id = _submit_and_wait(client)
    url = _media_url(client, job_id)
    manager._jobs[job_id].file_path.unlink()

    assert client.get(url).status_code == 404


# ---------------------------------------------------------------------------
# Numeric seek offsets alongside the display strings
# ---------------------------------------------------------------------------


def test_transcript_rows_carry_numeric_seconds() -> None:
    """The player seeks by number, so rows expose parsed offsets.

    The stored ``start``/``end`` are ``str(timedelta)`` for display; a media
    element needs ``currentTime`` in seconds, and parsing that in the browser
    would duplicate a parser the backend already has.
    """
    segment = _normalize_transcript_row({"start": "0:00:10", "end": "0:01:05", "text": "hi"})
    assert segment.start_seconds == 10.0
    assert segment.end_seconds == 65.0
    # The display strings are untouched.
    assert segment.start == "0:00:10"


def test_transcript_rows_handle_recordings_past_a_day() -> None:
    """``timedelta`` prefixes a day component, which must not break seeking."""
    segment = _normalize_transcript_row({"start": "1 day, 0:00:01", "end": "1 day, 0:00:02", "text": "x"})
    assert segment.start_seconds == 86401.0


def test_transcript_rows_degrade_to_none_on_unparseable_times() -> None:
    """A malformed timestamp costs that row its seek, not the whole job."""
    segment = _normalize_transcript_row({"start": "not a time", "end": None, "text": "x"})
    assert segment.start_seconds is None
    assert segment.end_seconds is None
    assert segment.text == "x"
