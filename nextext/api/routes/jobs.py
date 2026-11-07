"""Job lifecycle routes under ``/api/v1/jobs``.

Per-route ownership checks enforce the privacy boundary established by
:func:`nextext.api.identity.resolve_principal`: any read or mutation on a
job other than the caller's own is reported as ``404 Not Found`` so the
existence of the row never leaks across owners.

Every route that hands out, stores or erases personal data writes one
data-protection audit event (:mod:`nextext.api.audit`) once the data is
fetched and before it is returned; an attempt on another owner's job is
recorded as ``denied``. The progress streams carry no personal data and are
not recorded. ``docs/audit.md`` lists every route and what it records.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
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
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from nextext.api.artifacts import (
    BATCH_ARTIFACTS,
    FACE_IMAGE_ARTIFACT,
    SUPPORTED_ARTIFACTS,
    render_artifact,
    render_batch_artifact,
)
from nextext.api.audit import RequestAudit, audited, media_audit
from nextext.api.identity import edge_secret_enforced, resolve_principal
from nextext.api.jobs import JobManager, JobState
from nextext.api.schemas import (
    JobCreateResponse,
    JobListItem,
    JobListResponse,
    JobOptions,
    JobSnapshot,
    JobStatus,
    MediaGrantResponse,
)
from nextext.utils.audit_event import Actor, Target

router = APIRouter(prefix="/jobs", tags=["jobs"])

# Job creation stores the upload and starts the model calls and the
# combination of their results, so a purpose required for any of those
# actions is required to create a job.
_CREATE_AUDIT = audited("collect", "job.create", gate=("ai_call", "combine"))
_LIST_AUDIT = audited("query", "job.list")
_GET_AUDIT = audited("query", "job.get", deferred=True)
_DELETE_AUDIT = audited("delete", "job.delete")
_ARTIFACT_AUDIT = audited("disclose", "artifact.download", path_criteria={"name": "artifact"})
_BATCH_AUDIT = audited("disclose", "artifact.download_batch", path_criteria={"name": "artifact"})
# A grant is what lets a media element stream the upload, so it needs the
# purpose a query needs; only its refusals are recorded.
_GRANT_AUDIT = audited("query", "media.grant")


def _download_recipient(name: str) -> str:
    """Name where a downloaded artifact goes, by its format.

    Args:
        name: Artifact name, e.g. ``transcript.csv``.

    Returns:
        str: ``download:<extension>``, e.g. ``download:csv``.
    """
    return f"download:{name.rsplit('.', 1)[-1]}"


def _job_target(job_id: str) -> Target:
    """Reference one job as an audit target.

    Args:
        job_id: Job identifier.

    Returns:
        Target: ``kind="job"``, that id, count 1.
    """
    return Target(kind="job", ids=[job_id], count=1)


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
        logger.warning(f"Invalid options JSON: {exc}")
        raise HTTPException(
            status_code=422,
            detail="Invalid request.",
        ) from exc
    try:
        return JobOptions.model_validate(payload)
    except ValidationError as exc:
        logger.warning(f"Invalid options: {exc.errors()}")
        raise HTTPException(
            status_code=422,
            detail="Invalid request.",
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
    owner_id: str = Depends(resolve_principal),
    audit: RequestAudit = Depends(_CREATE_AUDIT),  # noqa: B008 — FastAPI dependency marker
) -> JobCreateResponse:
    """Accept a multipart upload and queue a new pipeline job.

    The manager records the upload (``collect`` / ``job.create``) before the
    job exists; an upload rejected for its size, or failing while it is
    stored, is removed and recorded with ``outcome="error"``. A job asking for
    faces while ``NEXTEXT_FACES`` is off is refused with ``422`` before the
    upload is stored.

    Args:
        file: Multipart audio/video upload.
        options: JSON-encoded ``JobOptions`` form field.
        manager: Job manager dependency.
        owner_id: Resolved principal (owner identifier).
        audit: The request's audit context.

    Returns:
        JobCreateResponse: Identifier and timestamps for the new job.
    """
    parsed_options = _parse_options(options)
    if parsed_options.faces and manager.faces_mode == "off":
        # Refused before anything is stored, so nothing is collected.
        logger.warning("Faces requested while NEXTEXT_FACES is off; job refused.")
        raise HTTPException(status_code=422, detail="Invalid request.")
    raw_name = Path(file.filename or "upload").name or "upload"
    suffix = Path(raw_name).suffix
    file_path, _ = manager.allocate_tmpdir(suffix=suffix)
    try:
        digest = await _stream_upload_to_disk(file, file_path, _max_upload_bytes())
    except BaseException as exc:
        # Remove the half-written file, then record the failed collection.
        manager._unlink_job_dir(file_path)
        failure = "upload_too_large" if isinstance(exc, HTTPException) else "upload_failed"
        audit.emit(outcome="error", criteria={"files": 1, "failure": failure})
        raise
    state = await manager.create_job(
        owner_id=owner_id,
        file_name=raw_name,
        file_path=file_path,
        source_file_hash=digest,
        options=parsed_options,
        origin=audit.origin,
    )
    logger.info(
        "Accepted job {} ({} bytes).",
        state.job_id,
        file_path.stat().st_size,
    )
    return JobCreateResponse(
        job_id=state.job_id,
        status=state.status,
        created_at=state.created_at,
    )


@router.get("", response_model=JobListResponse)
async def list_jobs(
    manager: JobManager = Depends(get_job_manager),  # noqa: B008 — FastAPI dependency marker
    owner_id: str = Depends(resolve_principal),
    audit: RequestAudit = Depends(_LIST_AUDIT),  # noqa: B008 — FastAPI dependency marker
) -> JobListResponse:
    """Return the caller's in-memory jobs, newest first.

    Jobs live only in memory and are scoped by owner. The frontend calls
    this on load to re-discover its own jobs after a browser reload —
    re-attaching to any still running and re-rendering those finished.
    Recorded as ``query`` / ``job.list`` naming every job listed.

    Args:
        manager: Job manager dependency.
        owner_id: Resolved principal (owner identifier).
        audit: The request's audit context.

    Returns:
        JobListResponse: The caller's jobs.
    """
    states = await manager.list_for_owner(owner_id)
    items = [
        JobListItem(
            job_id=state.job_id,
            status=state.status,
            file_name=state.file_name,
            stage=state.stage,
            progress=state.progress,
            error=state.error,
            error_code=state.error_code,
            skipped=state.is_skipped,
            skip_reason_code=state.skip_reason_code,
            created_at=state.created_at,
            started_at=state.started_at,
            finished_at=state.finished_at,
            task=state.options.task,
        )
        for state in states
    ]
    audit.emit_ids("job", [state.job_id for state in states])
    return JobListResponse(jobs=items)


_BATCH_DOWNLOAD_NAMES: dict[str, str] = {
    "docint.jsonl": "nextext_docint.jsonl",
    "archive.zip": "nextext_batch.zip",
}


# Declared before ``/{job_id}`` so the literal ``batch`` prefix is matched
# ahead of any job-id pattern.
@router.get("/batch/{name}")
async def download_batch_artifact(
    name: str,
    manager: JobManager = Depends(get_job_manager),  # noqa: B008 — FastAPI dependency marker
    owner_id: str = Depends(resolve_principal),
    audit: RequestAudit = Depends(_BATCH_AUDIT),  # noqa: B008 — FastAPI dependency marker
) -> Response:
    """Return one batch artifact spanning all of the caller's completed jobs.

    Combines every completed job the caller owns into a single download:
    ``docint.jsonl`` concatenates each job's docint records; ``archive.zip``
    nests each job's outputs under its own folder. Owner-scoped — only the
    caller's own jobs are ever included, so the batch never leaks across
    owners. Recorded as ``disclose`` / ``artifact.download_batch`` naming the
    jobs the bundle was built from.

    Args:
        name: Batch artifact name (``docint.jsonl`` or ``archive.zip``).
        manager: Job manager dependency.
        owner_id: Resolved principal (owner identifier).
        audit: The request's audit context.

    Returns:
        Response: Binary payload with the appropriate media type.
    """
    if name not in BATCH_ARTIFACTS:
        raise HTTPException(
            status_code=404,
            detail=f"Unsupported batch artifact '{name}'.",
        )
    states = await manager.list_for_owner(owner_id)
    completed = [state for state in states if state.status == JobStatus.COMPLETED]
    job_ids = [state.job_id for state in completed]
    # Assembling a multi-file batch ZIP is CPU-bound (openpyxl XLSX + nested
    # zips); render it off the event loop so a large bundle does not stall SSE
    # streams, health checks, or other downloads while it builds.
    try:
        rendered = await run_in_threadpool(render_batch_artifact, completed, name)
    except Exception:
        audit.emit_ids(
            "job", job_ids, outcome="error", criteria={"artifact": name}, recipient=_download_recipient(name)
        )
        raise
    if rendered is None:
        raise HTTPException(
            status_code=404,
            detail=f"No completed jobs produced '{name}'.",
        )
    payload, content_type = rendered
    audit.emit_ids("job", job_ids, criteria={"artifact": name}, recipient=_download_recipient(name))
    download_name = _BATCH_DOWNLOAD_NAMES[name]
    quoted = urllib.parse.quote(download_name, safe="")
    return Response(
        content=payload,
        media_type=content_type,
        headers={
            "Content-Disposition": (f"attachment; filename=\"{download_name}\"; filename*=UTF-8''{quoted}"),
            "Cache-Control": "no-store",
        },
    )


# Declared before ``/{job_id}`` so the literal ``events`` segment is matched
# ahead of the job-id pattern (mirrors the ``/batch/{name}`` precedent above).
@router.get("/events")
async def stream_owner_events(
    manager: JobManager = Depends(get_job_manager),  # noqa: B008 — FastAPI dependency marker
    owner_id: str = Depends(resolve_principal),
) -> StreamingResponse:
    """Stream SSE events for every job the caller owns over one connection.

    A single browser opens exactly one of these streams for its whole session
    instead of one per job, so a large batch never approaches the browser's
    per-host connection limit. Events for jobs created after the stream opens
    are delivered live; each frame carries its ``job_id`` so the client routes
    it to the right job. Owner-scoped — only the caller's own jobs appear.

    Args:
        manager: Job manager dependency.
        owner_id: Resolved principal (owner identifier).

    Returns:
        StreamingResponse: ``text/event-stream`` that stays open until the
            client disconnects.
    """
    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return StreamingResponse(
        manager.subscribe_owner(owner_id),
        media_type="text/event-stream",
        headers=headers,
    )


async def _require_job(
    job_id: str,
    manager: JobManager,
    owner_id: str,
    audit: RequestAudit | None = None,
    criteria: dict[str, object] | None = None,
) -> JobState:
    """Fetch a job and verify the caller owns it, or 404.

    Cross-owner access deliberately returns ``404`` rather than ``403``
    so the existence of the row stays private. On an audited route the
    attempt is recorded as refused (``outcome="denied"``, naming the job);
    the caller cannot see that record.

    Args:
        job_id: Job identifier.
        manager: Job manager dependency.
        owner_id: Resolved principal (owner identifier).
        audit: The route's audit context, if the route is audited.
        criteria: The route's parameters, for the refusal.

    Returns:
        JobState: The matching state.

    Raises:
        HTTPException: 404 when the job does not exist or belongs to a
            different owner.
    """
    state = await manager.get(job_id)
    if state is None or state.owner_id != owner_id:
        if state is not None and audit is not None:
            audit.emit(outcome="denied", target=_job_target(job_id), criteria=criteria)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return state


@router.get("/{job_id}", response_model=JobSnapshot)
async def get_job(
    job_id: str,
    manager: JobManager = Depends(get_job_manager),  # noqa: B008 — FastAPI dependency marker
    owner_id: str = Depends(resolve_principal),
    audit: RequestAudit = Depends(_GET_AUDIT),  # noqa: B008 — FastAPI dependency marker
) -> JobSnapshot:
    """Return a point-in-time view of one job.

    A finished job's snapshot carries its result — transcript, translation,
    summary, entities, findings — and is recorded as ``query`` /
    ``job.get``; until then the snapshot is a status poll (status, progress,
    the caller's own options) and is not recorded.

    Args:
        job_id: Job identifier.
        manager: Job manager dependency.
        owner_id: Resolved principal (owner identifier).
        audit: The request's audit context.

    Returns:
        JobSnapshot: Current job state.
    """
    state = await _require_job(job_id, manager, owner_id, audit)
    snapshot = state.snapshot()
    if snapshot.result is not None:
        audit.require_purpose()
        audit.emit(target=_job_target(job_id))
    return snapshot


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(
    job_id: str,
    manager: JobManager = Depends(get_job_manager),  # noqa: B008 — FastAPI dependency marker
    owner_id: str = Depends(resolve_principal),
    audit: RequestAudit = Depends(_DELETE_AUDIT),  # noqa: B008 — FastAPI dependency marker
) -> Response:
    """Remove a job, its tempfile, and signal SSE subscribers.

    The manager records the erasure (``delete`` / ``job.delete``). The SPA's
    "clear" actions issue one such request per job.

    Args:
        job_id: Job identifier.
        manager: Job manager dependency.
        owner_id: Resolved principal (owner identifier).
        audit: The request's audit context.

    Returns:
        Response: Empty 204 response.
    """
    await _require_job(job_id, manager, owner_id, audit)
    removed = await manager.delete(job_id, origin=audit.origin, owner_id=owner_id)
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
    owner_id: str = Depends(resolve_principal),
) -> StreamingResponse:
    """Stream SSE events for one job.

    Args:
        job_id: Job identifier.
        manager: Job manager dependency.
        owner_id: Resolved principal (owner identifier).

    Returns:
        StreamingResponse: ``text/event-stream`` connection that closes when
            the job reaches a terminal state.
    """
    state = await _require_job(job_id, manager, owner_id)
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


# Containers the upload dropzone accepts that stdlib ``mimetypes`` does not
# know (or maps to a non-media type). The browser picks <video> vs <audio> from
# this header, so a wrong guess renders an unplayable element.
_MEDIA_TYPE_OVERRIDES: dict[str, str] = {
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
}


def _media_type_for(file_name: str) -> str:
    """Guess the media type to serve a job's upload under.

    Args:
        file_name (str): Original upload filename.

    Returns:
        str: A MIME type; ``application/octet-stream`` when unguessable.
    """
    suffix = Path(file_name).suffix.lower()
    if suffix in _MEDIA_TYPE_OVERRIDES:
        return _MEDIA_TYPE_OVERRIDES[suffix]
    guessed, _ = mimetypes.guess_type(file_name)
    return guessed or "application/octet-stream"


def _media_unavailable() -> HTTPException:
    """Return the one answer every refused media request gets.

    Returns:
        HTTPException: 404 with a generic detail — never 403, so a wrong
            token or grant cannot be used to probe which jobs exist.
    """
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media unavailable.")


def _media_actor(caller: Actor | None, holder: Actor, owner_id: str) -> Actor:
    """Return whom an authorized media request is recorded as.

    Args:
        caller: The request's resolved caller, if any.
        holder: Whom the capability was issued to: the job's creator for the
            per-job token, the requester for a grant.
        owner_id: The job's owner.

    Returns:
        Actor: The caller, marked as acting on the owner's data when it is
            someone else; the holder when no caller resolved.
    """
    if caller is None:
        return holder
    if caller.user == owner_id:
        return caller
    return Actor(caller.user, caller.via, on_behalf_of=owner_id)


@router.post(
    "/{job_id}/media-grant",
    response_model=MediaGrantResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_media_grant(
    job_id: str,
    response: Response,
    manager: JobManager = Depends(get_job_manager),  # noqa: B008 — FastAPI dependency marker
    owner_id: str = Depends(resolve_principal),
    audit: RequestAudit = Depends(_GRANT_AUDIT),  # noqa: B008 — FastAPI dependency marker
) -> MediaGrantResponse:
    """Issue a playback grant for a job's upload.

    A media element sends neither the identity header nor ``X-Audit-Purpose``,
    so the SPA asks for a grant first, through its purpose wrapper, and plays
    the URL it gets back. The grant binds this request's caller, stated
    purpose and request id to a fresh token, which every range request of the
    playback presents until it expires (``NEXTEXT_MEDIA_GRANT_TTL_S``); each
    stream is recorded with that purpose (``query`` / ``media.stream``).

    Owner-scoped like the snapshot: another owner's job answers 404 and the
    attempt is recorded as denied (``query`` / ``media.grant``), as is a
    request that lacks the purpose ``AUDIT_REQUIRE_PURPOSE`` asks for (428).
    Issuing the grant hands out no personal data and is not recorded.

    Args:
        job_id: Job identifier.
        response: The response, to keep the token out of any cache.
        manager: Job manager dependency.
        owner_id: Resolved principal (owner identifier).
        audit: The request's audit context.

    Returns:
        MediaGrantResponse: The media URL carrying the grant, and its expiry.

    Raises:
        HTTPException: 404 for an unknown or foreign job, or an upload that is
            no longer on disk.
    """
    state = await _require_job(job_id, manager, owner_id, audit)
    if not state.file_path.exists():
        raise _media_unavailable()
    token, grant = manager.issue_media_grant(state, audit.origin)
    response.headers["Cache-Control"] = "no-store"
    return MediaGrantResponse(media_url=f"/api/v1/jobs/{job_id}/media?grant={token}", expires_at=grant.expires_at)


async def _authorize_token(
    job_id: str, token: str, manager: JobManager, audit: RequestAudit
) -> tuple[JobState, RequestAudit]:
    """Authorize a media request by the job's own capability token.

    Args:
        job_id: Job identifier from the path.
        token: The presented per-job token.
        manager: The job manager.
        audit: The request's audit context.

    Returns:
        tuple[JobState, RequestAudit]: The job, and the audit context with the
            actor bound.

    Raises:
        HTTPException: 404 for an unknown job or wrong token, a missing upload,
            or (edge secret set) a caller without a verified identity.
    """
    state = await manager.get_by_media_token(job_id, token)
    if state is None or not state.file_path.exists():
        raise _media_unavailable()
    if audit.actor is None and edge_secret_enforced():
        raise _media_unavailable()
    return state, audit.bind(_media_actor(audit.actor, state.audit.origin.actor, state.owner_id))


async def _authorize_grant(
    job_id: str, token: str, manager: JobManager, audit: RequestAudit, criteria: dict[str, object]
) -> tuple[JobState, RequestAudit]:
    """Authorize a media request by a playback grant.

    Adds the grant's id to ``criteria``. With the edge secret set, the request
    must come from the identity the grant was issued to: another verified
    identity is refused and recorded as denied; a request without a verified
    identity is refused unrecorded, like an unknown or expired grant.

    Args:
        job_id: Job identifier from the path.
        token: The presented grant token.
        manager: The job manager.
        audit: The request's audit context.
        criteria: The event's criteria, extended in place.

    Returns:
        tuple[JobState, RequestAudit]: The job, and the audit context carrying
            the actor and the grant request's purpose and request id.

    Raises:
        HTTPException: 404 for an unknown, expired or foreign grant, a missing
            upload, or (edge secret set) a caller who is not the grant's.
    """
    found = await manager.get_by_media_grant(job_id, token)
    if found is None or not found[0].file_path.exists():
        raise _media_unavailable()
    state, grant = found
    criteria["grant"] = grant.grant_id
    holder = grant.origin.actor
    caller = audit.actor
    if edge_secret_enforced():
        if caller is None:
            raise _media_unavailable()
        if (caller.user, caller.via) != (holder.user, holder.via):
            audit.emit(outcome="denied", target=_job_target(job_id), criteria=criteria)
            raise _media_unavailable()
    return state, audit.authorized_by(grant.origin, _media_actor(caller, holder, state.owner_id))


@router.get("/{job_id}/media")
async def stream_media(
    job_id: str,
    request: Request,
    token: str | None = Query(None, description="Per-job media capability token (the snapshot's `media_url`)."),
    grant: str | None = Query(None, description="Playback grant from `POST /jobs/{id}/media-grant`."),
    manager: JobManager = Depends(get_job_manager),  # noqa: B008 — FastAPI dependency marker
    audit: RequestAudit = Depends(media_audit),  # noqa: B008 — FastAPI dependency marker
) -> FileResponse:
    """Stream a job's original upload for in-browser playback.

    Deliberately **not** principal-scoped: a ``<video>`` / ``<audio>`` element
    cannot attach the trusted identity header, so an unguessable value in the
    URL is the authorization instead — exactly one of:

    * ``grant``: a playback grant (``POST /jobs/{id}/media-grant``), issued to
      a request that stated its purpose and valid for
      ``NEXTEXT_MEDIA_GRANT_TTL_S``. The SPA plays with these.
    * ``token``: the per-job capability from the owner-scoped snapshot
      (``JobResult.media_url``). It names no purpose, so it is refused with
      428 while ``AUDIT_REQUIRE_PURPOSE`` lists ``query``.

    Both die with the job (``DELETE /jobs/{id}``). They ride in the query
    string; the app's own nginx and uvicorn access logs drop the query string,
    so neither records them, though nginx's error log still does when it logs
    an error for the request.

    Served via ``FileResponse``, which answers ``Range`` requests with ``206``
    — that is what lets the player seek without re-fetching the whole
    recording.

    Every request that streams bytes — each range a player fetches — is
    recorded as ``query`` / ``media.stream``, never with the token or grant;
    a grant's id and purpose are recorded instead. The actor is the resolved
    caller, marked as acting on the owner's data when it is someone else.
    With ``NEXTEXT_EDGE_SECRET`` set, a request without a verified identity
    did not come through the gateway and is refused, and a grant plays only
    for the identity it was issued to; without it, a request without an
    identity is attributed to whom the token or grant was issued.

    Args:
        job_id (str): Job identifier from the path.
        request (Request): The incoming request, for its ``Range`` header.
        token (str | None): Capability token minted for this job.
        grant (str | None): Playback grant token.
        manager (JobManager): Injected job manager.
        audit (RequestAudit): The request's audit context.

    Returns:
        FileResponse: The upload bytes, ranged on request.

    Raises:
        HTTPException: 404 for an unknown job, a wrong token, an unknown or
            expired grant, neither or both given, an upload that is no longer
            on disk, or (edge secret set) an unverified caller or one the
            grant was not issued to; 428 when a stated purpose is required
            and missing.
    """
    criteria: dict[str, object] = {"ranged": "range" in request.headers}
    if grant is not None and token is None:
        state, audit = await _authorize_grant(job_id, grant, manager, audit, criteria)
    elif token is not None and grant is None:
        state, audit = await _authorize_token(job_id, token, manager, audit)
    else:
        raise _media_unavailable()
    audit.require_purpose()
    audit.emit(target=_job_target(job_id), criteria=criteria)
    return FileResponse(
        state.file_path,
        media_type=_media_type_for(state.file_name),
        # No `attachment` disposition: this is played in place, not saved.
        # `private` keeps shared caches out of it while letting the browser
        # reuse ranges it already fetched during a seek.
        headers={"Cache-Control": "private, max-age=0"},
    )


@router.get("/{job_id}/artifacts/{name}")
async def download_artifact(
    job_id: str,
    name: str,
    manager: JobManager = Depends(get_job_manager),  # noqa: B008 — FastAPI dependency marker
    owner_id: str = Depends(resolve_principal),
    audit: RequestAudit = Depends(_ARTIFACT_AUDIT),  # noqa: B008 — FastAPI dependency marker
) -> Response:
    """Return one artifact byte stream for a completed job.

    Recorded as ``disclose`` / ``artifact.download`` to
    ``download:<format>``.

    Args:
        job_id: Job identifier.
        name: Artifact name (e.g. ``transcript.csv``).
        manager: Job manager dependency.
        owner_id: Resolved principal (owner identifier).
        audit: The request's audit context.

    Returns:
        Response: Binary payload with the appropriate media type.
    """
    if name not in SUPPORTED_ARTIFACTS and not FACE_IMAGE_ARTIFACT.fullmatch(name):
        raise HTTPException(
            status_code=404,
            detail=f"Unsupported artifact '{name}'.",
        )
    criteria: dict[str, object] = {"artifact": name}
    state = await _require_job(job_id, manager, owner_id, audit, criteria)
    if state.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"Job '{job_id}' is in status '{state.status}'; artifacts unavailable.",
        )
    # Rendering (esp. archive.zip / XLSX) is CPU-bound; keep it off the event
    # loop so a large artifact does not block concurrent requests.
    try:
        rendered = await run_in_threadpool(render_artifact, state, name)
    except Exception:
        audit.emit(outcome="error", target=_job_target(job_id), criteria=criteria, recipient=_download_recipient(name))
        raise
    if rendered is None:
        raise HTTPException(
            status_code=404,
            detail=f"Artifact '{name}' was not produced by job '{job_id}'.",
        )
    payload, content_type = rendered
    audit.emit(target=_job_target(job_id), criteria=criteria, recipient=_download_recipient(name))
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
