"""Helpers for tests that check the data-protection audit events Nextext writes."""

from __future__ import annotations

import io
import json
from typing import Any

from nextext.audit import SERVICE, AuditSink, JobAudit, Origin
from nextext.utils.audit_event import Actor, AuditEmitter, verify_chain

# Fields that would carry content; no Nextext event may have any of them.
CONTENT_KEYS = ("text", "transcript", "prompt", "answer", "summary", "caption", "file_name", "filename", "token")


class AuditCapture:
    """Audit emitters writing to one buffer, and helpers to read what they wrote.

    Each emitter is its own chain, as each process is; the verifier tells
    them apart by instance.
    """

    def __init__(self) -> None:
        """Create the buffer; emitters are made on demand."""
        self.stream = io.StringIO()
        self.emitters: list[AuditEmitter] = []

    def factory(self, _output: str | None) -> AuditEmitter:
        """Stand in for :func:`nextext.audit.build_emitter`: a new emitter on the buffer.

        Args:
            _output: The ``AUDIT_OUTPUT`` mode, ignored.

        Returns:
            AuditEmitter: A new, unstarted emitter (heartbeat off).
        """
        emitter = AuditEmitter(SERVICE, "0.0.0+test", stream=self.stream, heartbeat_interval_s=0)
        self.emitters.append(emitter)
        return emitter

    def started(self) -> AuditEmitter:
        """Make a new emitter on the buffer and start it.

        Returns:
            AuditEmitter: The started emitter.
        """
        emitter = self.factory(None)
        emitter.start()
        return emitter

    @property
    def events(self) -> list[dict[str, Any]]:
        """Every event written so far.

        Returns:
            list[dict[str, Any]]: The parsed lines, in order.
        """
        return [json.loads(line) for line in self.stream.getvalue().splitlines()]

    def of(self, action: str | None = None, operation: str | None = None) -> list[dict[str, Any]]:
        """Return the non-lifecycle events, optionally of one action/operation.

        Args:
            action: Keep only this action.
            operation: Keep only this operation.

        Returns:
            list[dict[str, Any]]: The matching events, in order.
        """
        return [
            event
            for event in self.events
            if event["action"] != "lifecycle"
            and (action is None or event["action"] == action)
            and (operation is None or event["operation"] == operation)
        ]

    def one(self, action: str, operation: str) -> dict[str, Any]:
        """Return the single event of an action/operation, failing otherwise.

        Args:
            action: The action.
            operation: The operation.

        Returns:
            dict[str, Any]: The event.
        """
        found = self.of(action, operation)
        assert len(found) == 1, f"expected one {action}/{operation} event, got {len(found)}"
        return found[0]

    def assert_chain_ok(self) -> None:
        """Assert the captured chain verifies without problems."""
        report = verify_chain(self.events)
        assert report.ok, report.problems


def assert_no_content(event: dict[str, Any], *values: str) -> None:
    """Assert an event carries no content field and none of the given values.

    Args:
        event: An audit event.
        *values: Strings that must not appear anywhere in it (synthetic
            transcript text, file names, tokens).
    """
    line = json.dumps(event, ensure_ascii=False)
    for key in CONTENT_KEYS:
        assert f'"{key}":' not in line, f"content field {key!r} in {event['operation']}"
    for value in values:
        assert value not in line, f"{value!r} leaked into {event['operation']}"


def capturing_job_audit(job_id: str = "0" * 32, owner: str = "owner-example") -> tuple[JobAudit, AuditCapture]:
    """Build a job's audit trail on a fresh, started capturing emitter.

    Args:
        job_id: The job id.
        owner: The user the job is attributed to (via ``edge``).

    Returns:
        tuple[JobAudit, AuditCapture]: The trail and the capture of its events.
    """
    capture = AuditCapture()
    audit = JobAudit(AuditSink(capture.started()), job_id, Origin(actor=Actor(owner, "edge")))
    return audit, capture


def job_audit(job_id: str = "0" * 32, owner: str = "owner-example") -> JobAudit:
    """Build a job's audit trail for tests that do not read its events.

    Args:
        job_id: The job id.
        owner: The user the job is attributed to.

    Returns:
        JobAudit: The trail.
    """
    return capturing_job_audit(job_id, owner)[0]


def edge_origin(owner: str = "owner-example") -> Origin:
    """The origin of a request from ``owner`` through the edge gateway.

    Args:
        owner: The user.

    Returns:
        Origin: ``Actor(owner, "edge")``, no purpose, no request id.
    """
    return Origin(actor=Actor(owner, "edge"))
