"""HTTP client used by the Streamlit frontend to talk to the FastAPI backend."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

import httpx
from loguru import logger

DEFAULT_BACKEND_URL = "http://backend:8000"
OWNER_HEADER = "X-Owner-Id"


@dataclass(frozen=True)
class StageEvent:
    """One SSE event yielded by :meth:`BackendClient.subscribe_events`."""

    name: str
    data: dict[str, Any]


class BackendClient:
    """Thin HTTP client for the Nextext backend.

    Attributes:
        base_url: Backend root URL, e.g. ``http://backend:8000``.
        client: Backing :class:`httpx.Client` (kept open for the session).
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: httpx.Timeout | None = None,
        transport: httpx.BaseTransport | None = None,
        owner_id: str | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            base_url: Backend root URL. Defaults to ``BACKEND_HOST`` env var
                or ``http://backend:8000`` when unset.
            timeout: Connection / read / write timeouts. Defaults to a
                long-lived configuration suited to long uploads and streamed
                events.
            transport: Optional transport (mainly used by tests).
            owner_id: Per-browser identifier sent in the ``X-Owner-Id``
                header on every request. The backend uses this to scope
                persistent rows. Sourced from the browser's
                ``localStorage`` in production; tests pass a stable value.
        """
        self.base_url = (base_url or os.getenv("BACKEND_HOST") or DEFAULT_BACKEND_URL).rstrip("/")
        self.owner_id = owner_id
        headers: dict[str, str] = {}
        if owner_id:
            headers[OWNER_HEADER] = owner_id
        self.client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout or httpx.Timeout(connect=10.0, read=None, write=300.0, pool=10.0),
            transport=transport,
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self.client.close()

    def __enter__(self) -> BackendClient:
        """Support ``with`` blocks for callers that want explicit cleanup.

        Returns:
            BackendClient: ``self``.
        """
        return self

    def __exit__(self, *exc: Any) -> None:
        """Close the underlying client on exit."""
        self.close()

    # ------------------------------------------------------------------ jobs
    def submit_job(
        self,
        file_name: str,
        payload: bytes,
        options: dict[str, Any],
    ) -> str:
        """Submit a file to the backend and return its job id.

        Args:
            file_name: Display name of the file.
            payload: Raw file bytes.
            options: Pipeline options dict matching :class:`JobOptions`.

        Returns:
            str: The new job's id.
        """
        files = {"file": (file_name, payload)}
        data = {"options": json.dumps(options)}
        response = self.client.post("/api/v1/jobs", files=files, data=data)
        response.raise_for_status()
        return cast(str, response.json()["job_id"])

    def subscribe_events(self, job_id: str) -> Iterator[StageEvent]:
        """Stream SSE events for a job until it terminates.

        Args:
            job_id: Job identifier.

        Yields:
            StageEvent: Parsed events in arrival order.
        """
        with self.client.stream(
            "GET",
            f"/api/v1/jobs/{job_id}/events",
            timeout=httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0),
        ) as response:
            response.raise_for_status()
            event_name = ""
            data_lines: list[str] = []
            for line in response.iter_lines():
                if line.startswith(":"):
                    # Heartbeat comment — ignore.
                    continue
                if line == "":
                    if event_name and data_lines:
                        try:
                            payload = json.loads("\n".join(data_lines))
                        except json.JSONDecodeError:
                            logger.warning("Discarding malformed SSE payload for {}.", job_id)
                            payload = {}
                        yield StageEvent(name=event_name, data=payload)
                    event_name = ""
                    data_lines = []
                    continue
                if line.startswith("event:"):
                    event_name = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:") :].strip())

    def get_snapshot(self, job_id: str) -> dict[str, Any]:
        """Return the snapshot dict for a job.

        Args:
            job_id: Job identifier.

        Returns:
            dict[str, Any]: The deserialized snapshot payload.
        """
        response = self.client.get(f"/api/v1/jobs/{job_id}")
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def download_artifact(
        self,
        job_id: str,
        name: str,
    ) -> tuple[bytes, str]:
        """Download one artifact for a completed job.

        Args:
            job_id: Job identifier.
            name: Artifact name (e.g. ``transcript.csv``).

        Returns:
            tuple[bytes, str]: ``(payload, content_type)``.
        """
        response = self.client.get(f"/api/v1/jobs/{job_id}/artifacts/{name}")
        response.raise_for_status()
        return response.content, response.headers.get("content-type", "")

    def delete_job(self, job_id: str) -> None:
        """Delete a job and free its server-side temp files.

        Args:
            job_id: Job identifier.
        """
        response = self.client.delete(f"/api/v1/jobs/{job_id}")
        if response.status_code not in (204, 404):
            response.raise_for_status()

    def list_jobs(self) -> list[dict[str, Any]]:
        """Return the caller's persistent jobs, newest first.

        Returns:
            list[dict[str, Any]]: One entry per saved job. The list is
                empty when the caller has never opted in to persistence.
        """
        response = self.client.get("/api/v1/jobs")
        response.raise_for_status()
        body = response.json()
        return list(body.get("jobs", []))

    # ----------------------------------------------------------------- meta
    def get_health(self) -> dict[str, Any]:
        """Return the backend health payload.

        Returns:
            dict[str, Any]: Decoded JSON body.
        """
        response = self.client.get("/api/v1/health")
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def get_languages(self) -> dict[str, list[dict[str, str]]]:
        """Return the language mappings used by the frontend dropdowns.

        Returns:
            dict[str, list[dict[str, str]]]: ``{"whisper": [...], "target": [...]}``.
        """
        response = self.client.get("/api/v1/languages")
        response.raise_for_status()
        return cast(dict[str, list[dict[str, str]]], response.json())
