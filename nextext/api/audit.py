"""Request-side audit: who asks, for what stated purpose, under which request id.

Every audited route takes a :class:`RequestAudit` through :func:`audited` (or
:func:`media_audit` for the token-authorized media route). The dependency

* resolves the caller (401 when there is none — such requests are not
  audited: they have no actor);
* reads the stated purpose from ``X-Audit-Purpose`` (400 with a generic
  message when the header is malformed; nothing is recorded then);
* keeps the gateway's ``X-Request-ID`` when it looks like one;
* refuses a request whose action needs a stated purpose
  (``AUDIT_REQUIRE_PURPOSE``) but carries none: it records the refusal
  (``outcome="denied"``) and answers 428.

The route writes the event itself, once the data is fetched and before the
response is returned, with :meth:`RequestAudit.emit`. A write that fails
raises, and the route answers the generic 500 without the data.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace

from fastapi import Depends, HTTPException, Request, status

from nextext.api.identity import Caller, resolve_caller, resolve_caller_optional
from nextext.audit import Origin, emit_with_ids
from nextext.utils.audit_event import Actor, AuditEmitter, Purpose, Target
from nextext.utils.audit_runtime import (
    PURPOSE_HEADER,
    PURPOSE_REQUIRED_STATUS,
    REQUEST_ID_HEADER,
    PurposeError,
    clean_request_id,
    parse_purpose_header,
)

__all__ = ["PURPOSE_INVALID_DETAIL", "PURPOSE_REQUIRED_DETAIL", "RequestAudit", "audited", "media_audit"]

PURPOSE_INVALID_DETAIL = "Invalid audit purpose."
PURPOSE_REQUIRED_DETAIL = "This request needs a stated purpose."


@dataclass(frozen=True)
class RequestAudit:
    """The audit context of one request, and the writer of its event.

    Attributes:
        emitter: The process's audit emitter.
        action: The event's action, e.g. ``query``.
        operation: The event's operation, e.g. ``job.get``.
        gate: The actions whose purpose requirement applies to the request —
            its own action, plus those of the background work it starts.
        required: The actions ``AUDIT_REQUIRE_PURPOSE`` lists.
        actor: The caller, or ``None`` on the media route until the route
            binds one.
        purpose: The stated purpose, or ``None`` when none was stated.
        request_id: The gateway's correlation id, when it looks like one.
    """

    emitter: AuditEmitter
    action: str
    operation: str
    gate: frozenset[str]
    required: frozenset[str]
    actor: Actor | None
    purpose: Purpose | None
    request_id: str | None

    def bind(self, actor: Actor) -> RequestAudit:
        """Return a copy that records ``actor`` as the caller.

        Args:
            actor: The caller to record.

        Returns:
            RequestAudit: The bound copy.
        """
        return replace(self, actor=actor)

    @property
    def origin(self) -> Origin:
        """The request as the origin of the background work it starts.

        Returns:
            Origin: The actor, purpose and request id.

        Raises:
            RuntimeError: If no actor is bound.
        """
        return Origin(actor=self._actor(), purpose=self.purpose, request_id=self.request_id)

    def _actor(self) -> Actor:
        """Return the bound actor.

        Returns:
            Actor: The caller.

        Raises:
            RuntimeError: If no actor is bound.
        """
        if self.actor is None:
            raise RuntimeError("audited request has no actor")
        return self.actor

    def require_purpose(self, criteria: Mapping[str, object] | None = None) -> None:
        """Refuse the request when its action needs a stated purpose it lacks.

        Records the refusal (``outcome="denied"``, no target) before raising.

        Args:
            criteria: The request's parameters, for the refusal event.

        Raises:
            HTTPException: 428 when a purpose is required and none was stated.
        """
        if self.purpose is None and self.gate & self.required:
            self.emit(outcome="denied", criteria=criteria)
            raise HTTPException(status_code=PURPOSE_REQUIRED_STATUS, detail=PURPOSE_REQUIRED_DETAIL)

    def emit(
        self,
        *,
        outcome: str = "ok",
        target: Target | None = None,
        criteria: Mapping[str, object] | None = None,
        recipient: str | None = None,
    ) -> dict[str, object]:
        """Write the request's event.

        Args:
            outcome: ``ok``, ``denied`` or ``error``.
            target: What the request touched, by reference.
            criteria: The request's parameters — never content.
            recipient: Where data went, for a disclosure.

        Returns:
            dict[str, object]: The event as written.
        """
        return self.emitter.emit(
            self.action,
            self.operation,
            self._actor(),
            outcome=outcome,
            request_id=self.request_id,
            purpose=self.purpose,
            target=target,
            criteria=criteria,
            recipient=recipient,
        )

    def emit_ids(
        self,
        kind: str,
        ids: Sequence[str],
        *,
        outcome: str = "ok",
        criteria: Mapping[str, object] | None = None,
        recipient: str | None = None,
    ) -> dict[str, object]:
        """Write the request's event naming any number of ids (continuations past 200).

        Args:
            kind: The target kind of every id.
            ids: The ids touched.
            outcome: ``ok``, ``denied`` or ``error``.
            criteria: The request's parameters.
            recipient: Where data went, for a disclosure.

        Returns:
            dict[str, object]: The first event as written.
        """
        return emit_with_ids(
            self.emitter,
            self.action,
            self.operation,
            self._actor(),
            kind=kind,
            ids=ids,
            outcome=outcome,
            request_id=self.request_id,
            purpose=self.purpose,
            criteria=criteria,
            recipient=recipient,
        )


def _app_audit_state(request: Request) -> tuple[AuditEmitter, frozenset[str]]:
    """Return the emitter and required actions the lifespan put on the app.

    Args:
        request: The incoming request.

    Returns:
        tuple[AuditEmitter, frozenset[str]]: The emitter and the actions that
            need a stated purpose.

    Raises:
        RuntimeError: If the app was started without its audit state.
    """
    emitter = getattr(request.app.state, "audit_emitter", None)
    required = getattr(request.app.state, "audit_required", None)
    if not isinstance(emitter, AuditEmitter) or not isinstance(required, frozenset):
        raise RuntimeError("The audit emitter is not initialized. Was the API started through its lifespan?")
    return emitter, required


def _stated_purpose(request: Request) -> Purpose | None:
    """Read the request's stated purpose from ``X-Audit-Purpose``.

    Args:
        request: The incoming request.

    Returns:
        Purpose | None: The purpose, or ``None`` when none was stated.

    Raises:
        HTTPException: 400 with a generic message when the header is malformed.
    """
    try:
        case_ref, reason = parse_purpose_header(request.headers.get(PURPOSE_HEADER))
    except PurposeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=PURPOSE_INVALID_DETAIL) from None
    if case_ref is None and reason is None:
        return None
    return Purpose(case_ref=case_ref, reason=reason)


def _request_audit(
    request: Request, action: str, operation: str, gate: frozenset[str], actor: Actor | None
) -> RequestAudit:
    """Build a request's audit context.

    Args:
        request: The incoming request.
        action: The event's action.
        operation: The event's operation.
        gate: The actions whose purpose requirement applies.
        actor: The caller, if resolved.

    Returns:
        RequestAudit: The context.
    """
    emitter, required = _app_audit_state(request)
    return RequestAudit(
        emitter=emitter,
        action=action,
        operation=operation,
        gate=gate,
        required=required,
        actor=actor,
        purpose=_stated_purpose(request),
        request_id=clean_request_id(request.headers.get(REQUEST_ID_HEADER)),
    )


def audited(
    action: str,
    operation: str,
    *,
    gate: Collection[str] = (),
    path_criteria: Mapping[str, str] | None = None,
    deferred: bool = False,
) -> Callable[[Request, Caller], RequestAudit]:
    """Build the audit dependency of one route.

    Args:
        action: The event's action, e.g. ``disclose``.
        operation: The event's operation, e.g. ``artifact.download``.
        gate: Further actions whose purpose requirement applies, e.g. the
            ``ai_call`` and ``combine`` a job creation starts.
        path_criteria: Path parameters to record in a refusal, as
            ``{path_param: criteria_key}``.
        deferred: When true, the route checks the purpose requirement itself
            (:meth:`RequestAudit.require_purpose`) once it knows the request
            touches personal data — e.g. a job snapshot, which is only a
            status poll until the job has finished.

    Returns:
        Callable[[Request, Caller], RequestAudit]: The FastAPI dependency.
    """
    gate_actions = frozenset({action, *gate})
    path_keys = dict(path_criteria or {})

    def dependency(request: Request, caller: Caller = Depends(resolve_caller)) -> RequestAudit:  # noqa: B008 — FastAPI dependency marker
        """Resolve the request's audit context, refusing it when a purpose is missing.

        Args:
            request: The incoming request.
            caller: The resolved caller (401 when there is none).

        Returns:
            RequestAudit: The context, with the caller bound.
        """
        audit = _request_audit(request, action, operation, gate_actions, caller.actor)
        if not deferred:
            params = request.path_params
            audit.require_purpose({key: params[name] for name, key in path_keys.items() if name in params} or None)
        return audit

    return dependency


def media_audit(
    request: Request,
    caller: Caller | None = Depends(resolve_caller_optional),  # noqa: B008 — FastAPI dependency marker
) -> RequestAudit:
    """Audit dependency of the media route (``query`` / ``media.stream``).

    The route is authorized by its capability token, not by the caller, so
    the caller may be unresolved here, and the purpose requirement is checked
    by the route once it knows the job: the route binds the actor and calls
    :meth:`RequestAudit.require_purpose` itself.

    Args:
        request: The incoming request.
        caller: The resolved caller, if any.

    Returns:
        RequestAudit: The context; ``actor`` is ``None`` when no caller
            resolved.
    """
    return _request_audit(
        request,
        "query",
        "media.stream",
        frozenset({"query"}),
        None if caller is None else caller.actor,
    )
