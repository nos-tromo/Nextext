"""Job lifecycle routes: ``POST``, ``GET``, ``DELETE`` under ``/api/v1/jobs``.

The SSE event stream and per-artifact downloads are wired in by step 4.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import ValidationError

from nextext.api.artifacts import SUPPORTED_ARTIFACTS, render_artifact
from nextext.api.jobs import JobManager, JobState
from nextext.api.schemas import JobCreateResponse, JobOptions, JobSnapshot, JobStatus

router = APIRouter(prefix="/jobs", tags=["jobs"])

_DEFAULT_MAX_UPLOAD_MB = 8192
_CHUNK_SIZE = 1 << 20  # 1 MiB — mirrors the Streamlit upload hash loop.
_SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_filename(stem: str, fallback: str = "result") -> str:
    """Restrict a filename component to characters safe in HTTP headers.

    Strips CRLF, double quotes, and any non-``[A-Za-z0-9._-]`` character so
    the result can be interpolated into ``Content-Disposition`` without
    enabling response splitting (CWE-93) or header injection (CWE-113).

    Args:
        stem: Raw filename component from a user-controlled source.
        fallback: Value to return when ``stem`` reduces to an empty string.

    Returns:
        str: A safe filename component.
    """
    cleaned = _SAFE_FILENAME_PATTERN.sub("_", stem).strip("._-")
    return cleaned or fallback


def get_job_manager(request: Request) -> JobManager:
    """FastAPI dependency that returns the per-app ``JobManager`` singleton.

    Args:
        request: The incoming request, used to reach ``app.state``.

    Returns:
        JobManager: The shared manager instance configured at startup.
    """
    manager = getattr(request.app.state, "job_manager", None)
    if not isinstance(manager, JobManager):
        raise RuntimeError(
            "JobManager has not been initialized. Was the API started through `create_app` + the lifespan hook?"
        )
    return manager


def _max_upload_bytes() -> int:
    """Resolve the per-upload byte cap from ``NEXTEXT_MAX_UPLOAD_MB``.

    Returns:
        int: The cap in bytes.
    """
    raw = os.getenv("NEXTEXT_MAX_UPLOAD_MB", str(_DEFAULT_MAX_UPLOAD_MB)).strip()
    try:
        mb = int(raw)
    except ValueError:
        mb = _DEFAULT_MAX_UPLOAD_MB
    return max(mb, 1) * (1 << 20)


def _parse_options(raw: str) -> JobOptions:
    """Parse the ``options`` form field into a validated ``JobOptions``.

    Args:
        raw: JSON string from the multipart ``options`` field.

    Returns:
        JobOptions: Validated options.

    Raises:
        HTTPException: When the field is missing, not JSON, or invalid.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid JSON in `options`: {exc.msg}",
        ) from exc
    try:
        return JobOptions.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(),
        ) from exc


async def _stream_upload_to_disk(
    upload: UploadFile,
    destination: Path,
    max_bytes: int,
) -> str:
    """Stream an upload to disk in 1 MiB chunks, computing its SHA-256 hash.

    Args:
        upload: The incoming ``UploadFile``.
        destination: Target path. Its parent directory must already exist.
        max_bytes: Hard cap on accepted bytes; raises 413 if exceeded.

    Returns:
        str: ``sha256:<hex>`` digest of the streamed payload.

    Raises:
        HTTPException: When the upload exceeds ``max_bytes``.
    """
    hasher = hashlib.sha256()
    received = 0
    with destination.open("wb") as out_file:
        while True:
            chunk = await upload.read(_CHUNK_SIZE)
            if not chunk:
                break
            received += len(chunk)
            if received > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(f"Upload exceeds the configured limit of {max_bytes // (1 << 20)} MiB."),
                )
            hasher.update(chunk)
            out_file.write(chunk)
    return f"sha256:{hasher.hexdigest()}"


@router.post(
    "",
    response_model=JobCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_job(
    file: UploadFile = File(..., description="Audio or video file to process."),  # noqa: B008 — FastAPI dependency marker
    options: str = Form(
        ...,
        description="JSON-encoded `JobOptions` payload.",
    ),
    manager: JobManager = Depends(get_job_manager),  # noqa: B008 — FastAPI dependency marker
) -> JobCreateResponse:
    """Accept a multipart upload and queue a new pipeline job.

    Args:
        file: Multipart audio/video upload.
        options: JSON-encoded ``JobOptions`` form field.
        manager: Job manager dependency.

    Returns:
        JobCreateResponse: Identifier and timestamps for the new job.
    """
    parsed_options = _parse_options(options)
    raw_name = Path(file.filename or "upload").name or "upload"
    suffix = Path(raw_name).suffix
    file_path, _ = manager.allocate_tmpdir(suffix=suffix)
    try:
        digest = await _stream_upload_to_disk(file, file_path, _max_upload_bytes())
    except HTTPException:
        # Clean up the half-written file before re-raising.
        manager._unlink_job_dir(file_path)
        raise
    state = await manager.create_job(
        file_name=raw_name,
        file_path=file_path,
        source_file_hash=digest,
        options=parsed_options,
    )
    logger.info("Accepted job {} ({} bytes).", state.job_id, file_path.stat().st_size)
    return JobCreateResponse(
        job_id=state.job_id,
        status=state.status,
        created_at=state.created_at,
    )


async def _require_job(job_id: str, manager: JobManager) -> JobState:
    """Fetch a job by id or raise 404.

    Args:
        job_id: Job identifier.
        manager: Job manager dependency.

    Returns:
        JobState: The matching state.

    Raises:
        HTTPException: 404 when the job does not exist.
    """
    state = await manager.get(job_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return state


@router.get("/{job_id}", response_model=JobSnapshot)
async def get_job(
    job_id: str,
    manager: JobManager = Depends(get_job_manager),  # noqa: B008 — FastAPI dependency marker
) -> JobSnapshot:
    """Return a point-in-time view of one job.

    Args:
        job_id: Job identifier.
        manager: Job manager dependency.

    Returns:
        JobSnapshot: Current job state.
    """
    state = await _require_job(job_id, manager)
    return state.snapshot()


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(
    job_id: str,
    manager: JobManager = Depends(get_job_manager),  # noqa: B008 — FastAPI dependency marker
) -> Response:
    """Remove a job, its tempfile, and signal SSE subscribers.

    Args:
        job_id: Job identifier.
        manager: Job manager dependency.

    Returns:
        Response: Empty 204 response.
    """
    removed = await manager.delete(job_id)
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{job_id}/events")
async def stream_job_events(
    job_id: str,
    manager: JobManager = Depends(get_job_manager),  # noqa: B008 — FastAPI dependency marker
) -> StreamingResponse:
    """Stream SSE events for one job.

    Args:
        job_id: Job identifier.
        manager: Job manager dependency.

    Returns:
        StreamingResponse: ``text/event-stream`` connection that closes when
            the job reaches a terminal state.
    """
    state = await _require_job(job_id, manager)
    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return StreamingResponse(
        manager.subscribe(state),
        media_type="text/event-stream",
        headers=headers,
    )


@router.get("/{job_id}/artifacts/{name}")
async def download_artifact(
    job_id: str,
    name: str,
    manager: JobManager = Depends(get_job_manager),  # noqa: B008 — FastAPI dependency marker
) -> Response:
    """Return one artifact byte stream for a completed job.

    Args:
        job_id: Job identifier.
        name: Artifact name (e.g. ``transcript.csv``).
        manager: Job manager dependency.

    Returns:
        Response: Binary payload with the appropriate media type.
    """
    if name not in SUPPORTED_ARTIFACTS:
        raise HTTPException(
            status_code=404,
            detail=f"Unsupported artifact '{name}'.",
        )
    state = await _require_job(job_id, manager)
    if state.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"Job '{job_id}' is in status '{state.status}'; artifacts unavailable.",
        )
    rendered = render_artifact(state, name)
    if rendered is None:
        raise HTTPException(
            status_code=404,
            detail=f"Artifact '{name}' was not produced by job '{job_id}'.",
        )
    payload, content_type = rendered
    safe_stem = _sanitize_filename(Path(state.file_name).stem)
    download_name = f"{safe_stem}_{name}"
    quoted = urllib.parse.quote(download_name, safe="")
    return Response(
        content=payload,
        media_type=content_type,
        headers={
            "Content-Disposition": (f"attachment; filename=\"{download_name}\"; filename*=UTF-8''{quoted}"),
            "Cache-Control": "no-store",
        },
    )
