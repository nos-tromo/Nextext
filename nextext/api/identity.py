"""Request-principal resolution for the Nextext API.

Identity is resolved per request, in this order:

1. A service token: ``Authorization: Bearer <token>`` matching an entry of
   ``NEXTEXT_SERVICE_TOKENS`` resolves to the service principal
   ``service:<name>``. The trusted header is ignored on such a request — a
   service cannot act as a user.
2. The trusted header (``NEXTEXT_AUTH_HEADER``, default ``X-Auth-User``). With
   ``NEXTEXT_EDGE_SECRET`` set it counts only on an edge-verified request, one
   whose ``X-Edge-Secret`` header matches the secret; on any other request it
   is ignored, with a warning, as if it were absent. A value in the service
   namespace (``service:…``) is ignored the same way on every request.
3. ``NEXTEXT_DEFAULT_IDENTITY``, the dev / header-less fallback, which may not
   lie in the service namespace either. It is honoured only while no edge
   secret is set: with both set it is ignored (startup warns once), so in the
   production shape a request without a verified identity gets 401.
4. Otherwise fail closed with HTTP 401.

Each source also says how the identity was established — ``service``,
``edge`` (the trusted header) or ``system`` (the default identity) — which
the data-protection audit log records as the actor's ``via``
(:class:`Caller`).

The React frontend carries a stable per-browser identifier in its URL
(``?owner=<id>``) and sends it under the trusted header, so two browsers stay
isolated for the purpose of scoping their in-memory jobs; behind the edge
gateway, the gateway replaces it with the signed-in user.

Trust model: only the edge gateway holds the edge secret, and it sends it on
every request it proxies. With the secret set, the trusted header is
therefore believed only from the gateway — any other caller that can reach
the backend, directly on ``inference-net`` or through the frontend's proxy,
cannot assert an identity with it. A service that calls the API directly
authenticates with its own token and acts only as itself, in a namespace no
user can enter: a user named like a service (``docint``) is a different
principal from the service (``service:docint``) and cannot see its jobs,
which hold other users' media and transcripts. With the secret
unset, the trusted header is believed from any caller that can reach the
backend, and startup warns about it unless ``NEXTEXT_DEFAULT_IDENTITY`` marks
a dev setup. Secrets and tokens are compared in constant time and never
logged.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from urllib.parse import quote

from fastapi import Depends, HTTPException, Request, status
from loguru import logger

from nextext.utils.audit_event import Actor
from nextext.utils.env_cfg import SERVICE_PRINCIPAL_PREFIX, PrincipalConfig, load_principal_env

EDGE_SECRET_HEADER = "X-Edge-Secret"


@dataclass(frozen=True)
class Caller:
    """The principal a request resolved to, and how it was established.

    Attributes:
        principal: The owner id that scopes the caller's jobs.
        via: ``service`` (a service token), ``edge`` (the trusted header) or
            ``system`` (``NEXTEXT_DEFAULT_IDENTITY``, the dev fallback).
    """

    principal: str
    via: str

    @property
    def actor(self) -> Actor:
        """The caller as the audit log records it.

        Returns:
            Actor: ``Actor(principal, via)``.
        """
        return Actor(self.principal, self.via)


def _matches(received: str, expected: str) -> bool:
    """Compare a received credential with a configured one in constant time.

    Starlette decodes header values as latin-1, so encoding back recovers the
    bytes on the wire. Comparing bytes also keeps a non-ASCII header value
    from raising, which ``hmac.compare_digest`` does for ``str`` input.

    Args:
        received: The credential as read from a request header.
        expected: The configured secret or token.

    Returns:
        bool: ``True`` when both are byte-for-byte equal.
    """
    return hmac.compare_digest(received.encode("latin-1"), expected.encode())


def _headers_trusted(request: Request, cfg: PrincipalConfig) -> bool:
    """Report whether the request's identity headers may be believed.

    They are when no edge secret is configured, or when the request is
    edge-verified: its ``X-Edge-Secret`` header matches the secret. A missing
    header never matches.

    Args:
        request: The incoming request.
        cfg: The loaded principal configuration.

    Returns:
        bool: Whether the identity headers count.
    """
    if cfg.edge_secret is None:
        return True
    received = request.headers.get(EDGE_SECRET_HEADER)
    return received is not None and _matches(received, cfg.edge_secret)


def identity_headers_trusted(request: Request) -> bool:
    """Report whether the gateway-injected ``X-Auth-*`` headers may be believed.

    Args:
        request: The incoming request.

    Returns:
        bool: ``True`` when ``NEXTEXT_EDGE_SECRET`` is unset or the request
            is edge-verified.
    """
    return _headers_trusted(request, load_principal_env())


def edge_secret_enforced() -> bool:
    """Report whether the trusted header counts only from the edge gateway.

    Returns:
        bool: ``True`` when ``NEXTEXT_EDGE_SECRET`` is set.
    """
    return load_principal_env().edge_secret is not None


def _bearer_token(request: Request) -> str | None:
    """Return the credential of an ``Authorization: Bearer`` header.

    Args:
        request: The incoming request.

    Returns:
        str | None: The token, or ``None`` when the header is absent, uses
            another scheme, or carries no credential.
    """
    scheme, _, credential = request.headers.get("Authorization", "").strip().partition(" ")
    if scheme.lower() != "bearer":
        return None
    return credential.strip() or None


def _service_principal(request: Request, cfg: PrincipalConfig) -> str | None:
    """Return the principal of the service whose token the request presents.

    The presented token is compared with every configured token, in constant
    time and without stopping at a match, so the timing does not reveal which
    entry it matched.

    Args:
        request: The incoming request.
        cfg: The loaded principal configuration.

    Returns:
        str | None: ``service:<name>`` for the matching entry, or ``None``.
    """
    if not cfg.service_tokens:
        return None
    presented = _bearer_token(request)
    if presented is None:
        return None
    matched: str | None = None
    for name, token in cfg.service_tokens:
        if _matches(presented, token):
            matched = name
    return None if matched is None else f"{SERVICE_PRINCIPAL_PREFIX}{matched}"


def _log_path(request: Request) -> str:
    """Return the request path for a log line, without query string or host.

    Read from the ASGI scope rather than ``request.url``, which is rebuilt
    from the client's ``Host`` header, and percent-encoded as uvicorn's access
    log does, so a crafted path cannot forge log lines.

    Args:
        request: The incoming request.

    Returns:
        str: The percent-encoded request path.
    """
    return quote(str(request.scope.get("path", "")))


def _resolve(request: Request) -> Caller | None:
    """Resolve the calling principal without failing.

    Resolution order:

    1. A matching service token (``Authorization: Bearer``) →
       ``service:<name>``, via ``service``.
    2. The trusted header, if present and non-blank — only on an
       edge-verified request when ``NEXTEXT_EDGE_SECRET`` is set, and never
       with a value in the service namespace; otherwise it is ignored and the
       method and path are logged. Via ``edge``.
    3. The configured default identity, only while no edge secret is set.
       Via ``system``.

    Args:
        request: The incoming FastAPI/Starlette request.

    Returns:
        Caller | None: The resolved caller, or ``None`` when none of the
            sources applies.
    """
    cfg = load_principal_env()
    service = _service_principal(request, cfg)
    if service is not None:
        return Caller(principal=service, via="service")
    header_value = (request.headers.get(cfg.header_name) or "").strip()
    if header_value:
        if not _headers_trusted(request, cfg):
            reason = f"the request carries no valid {EDGE_SECRET_HEADER}"
        elif header_value.startswith(SERVICE_PRINCIPAL_PREFIX):
            reason = "its value lies in the reserved service namespace"
        else:
            return Caller(principal=header_value, via="edge")
        logger.warning(
            "Ignored the {} header on {} {}: {}.",
            cfg.header_name,
            request.method,
            _log_path(request),
            reason,
        )
    if cfg.default_identity and cfg.edge_secret is None:
        return Caller(principal=cfg.default_identity, via="system")
    return None


def resolve_caller(request: Request) -> Caller:
    """FastAPI dependency that resolves the calling principal, or fails with 401.

    See :func:`_resolve` for the resolution order.

    Args:
        request: The incoming FastAPI/Starlette request.

    Returns:
        Caller: The principal and how it was established.

    Raises:
        HTTPException: 401 when neither a service token, a trusted header,
            nor a usable default identity is available.
    """
    caller = _resolve(request)
    if caller is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authenticated principal.",
        )
    return caller


def resolve_caller_optional(request: Request) -> Caller | None:
    """FastAPI dependency that resolves the calling principal, if there is one.

    For the media route, which a ``<video>`` element reaches without the
    trusted header and which is authorized by its capability token instead.

    Args:
        request: The incoming FastAPI/Starlette request.

    Returns:
        Caller | None: The resolved caller, or ``None``.
    """
    return _resolve(request)


def resolve_principal(caller: Caller = Depends(resolve_caller)) -> str:  # noqa: B008 — FastAPI dependency marker
    """FastAPI dependency that resolves the calling principal (owner id).

    Args:
        caller: The resolved caller (:func:`resolve_caller`); FastAPI resolves
            it once per request, however many dependencies ask for it.

    Returns:
        str: The resolved principal identifier used to scope owned jobs.
    """
    return caller.principal


def check_principal_config() -> None:
    """Validate the identity configuration once, at startup.

    Loading it eagerly makes a malformed ``NEXTEXT_EDGE_SECRET`` or
    ``NEXTEXT_SERVICE_TOKENS`` stop the backend before it serves a request.
    In the production shape (no ``NEXTEXT_DEFAULT_IDENTITY``) without an edge
    secret, it also logs one warning: the trusted header is then believed
    from any caller. With both an edge secret and a default identity set, it
    warns that the default identity is ignored.

    Raises:
        PrincipalConfigError: When the identity configuration is malformed.
    """
    cfg = load_principal_env()
    if cfg.edge_secret is None and cfg.default_identity is None:
        logger.warning(
            "NEXTEXT_EDGE_SECRET is unset: the {} identity header is trusted from any caller that "
            "can reach the backend. Set NEXTEXT_EDGE_SECRET (see edge-plane docs/identity-contract.md).",
            cfg.header_name,
        )
    if cfg.edge_secret is not None and cfg.default_identity is not None:
        logger.warning(
            "NEXTEXT_DEFAULT_IDENTITY is ignored while NEXTEXT_EDGE_SECRET is set: a request without a "
            "verified identity gets 401. Unset NEXTEXT_DEFAULT_IDENTITY in this setup."
        )


__all__ = [
    "EDGE_SECRET_HEADER",
    "Caller",
    "check_principal_config",
    "edge_secret_enforced",
    "identity_headers_trusted",
    "resolve_caller",
    "resolve_caller_optional",
    "resolve_principal",
]
