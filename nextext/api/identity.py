"""Anonymous per-browser identity for the Nextext API.

The backend has no authentication; every request that reaches it is
trusted on the network level. The identity layer adds a single bit of
privacy plumbing on top of that: an opaque UUID cookie that lets the
backend tell two browsers apart so persistent jobs can be scoped to the
session that submitted them.

Design notes:

- The cookie value is a UUID4 hex string. It carries no PII and is not
  shared cross-host (``SameSite=Lax``), so it survives the typical
  refresh / back-button flow but does not leak via embeds or cross-site
  fetches.
- ``HttpOnly`` keeps the value out of ``document.cookie``, so a future
  XSS bug cannot exfiltrate it.
- ``Secure`` is configurable so local-development HTTP works; production
  deployments behind HTTPS should set ``NEXTEXT_SESSION_COOKIE_SECURE=1``.
- The cookie's ``Max-Age`` defaults to one year. Wiping site data
  produces a fresh identity, which is the correct privacy default — old
  persistent rows belong to the previous identity and are intentionally
  unreachable to the new one.
"""

from __future__ import annotations

import os
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

COOKIE_NAME = "nextext_session"
_DEFAULT_TTL_DAYS = 365
_MAX_TTL_DAYS = 3650  # 10 years — matches modern browser cookie cap.


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a truthy/falsy environment variable.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset.

    Returns:
        bool: ``True`` for ``1``/``true``/``yes``/``on``, ``False`` for the
            negative tokens, ``default`` for any other value.
    """
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _resolve_ttl_seconds() -> int:
    """Return the cookie ``Max-Age`` in seconds.

    Reads ``NEXTEXT_SESSION_COOKIE_TTL_DAYS`` (default 365 days, clamped
    to ``[1, _MAX_TTL_DAYS]``). The upper bound matches what modern
    browsers will accept; values above it provide no practical benefit
    and risk producing surprising configurations.

    Returns:
        int: Cookie lifetime in seconds.
    """
    raw = os.getenv("NEXTEXT_SESSION_COOKIE_TTL_DAYS", str(_DEFAULT_TTL_DAYS)).strip()
    try:
        days = int(raw)
    except ValueError:
        days = _DEFAULT_TTL_DAYS
    return max(1, min(days, _MAX_TTL_DAYS)) * 86400


def _is_valid_token(value: str | None) -> bool:
    """Return whether ``value`` parses as a hex-encoded UUID4.

    A strict parser keeps spoofed / malformed cookies from being
    propagated; we mint a fresh identity in that case so the user does
    not accidentally write rows with a garbage owner_id.

    Args:
        value: Raw cookie value (or ``None``).

    Returns:
        bool: ``True`` when ``value`` is a recognised UUID; ``False``
            otherwise.
    """
    if not value or len(value) != 32:
        return False
    try:
        uuid.UUID(hex=value)
    except (TypeError, ValueError):
        return False
    return True


def _mint_owner_id() -> str:
    """Generate a fresh opaque owner identifier.

    Returns:
        str: A 32-character hex-encoded UUID4.
    """
    return uuid.uuid4().hex


class IdentityMiddleware(BaseHTTPMiddleware):
    """Attach a per-browser ``owner_id`` to every request.

    The middleware reads the ``nextext_session`` cookie if present and
    valid; otherwise it mints a new UUID. The resolved value is attached
    to ``request.state.owner_id`` so the route layer can use it without
    re-parsing the cookie header. Freshly minted identities are written
    back via ``Set-Cookie`` on the outbound response.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        cookie_name: str = COOKIE_NAME,
        secure: bool | None = None,
        max_age_seconds: int | None = None,
    ) -> None:
        """Configure the middleware.

        Args:
            app: Wrapped ASGI application.
            cookie_name: Name of the session cookie.
            secure: Override for the ``Secure`` flag. ``None`` reads
                ``NEXTEXT_SESSION_COOKIE_SECURE`` from the environment.
            max_age_seconds: Override for the cookie ``Max-Age``. ``None``
                reads ``NEXTEXT_SESSION_COOKIE_TTL_DAYS``.
        """
        super().__init__(app)
        self._cookie_name = cookie_name
        self._secure = (
            secure if secure is not None else _env_bool("NEXTEXT_SESSION_COOKIE_SECURE")
        )
        self._max_age = max_age_seconds or _resolve_ttl_seconds()

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        """Inject ``owner_id`` and persist newly minted identities.

        Args:
            request: Incoming request.
            call_next: Downstream handler.

        Returns:
            Response: The downstream response, possibly with an added
                ``Set-Cookie`` header.
        """
        raw = request.cookies.get(self._cookie_name)
        if _is_valid_token(raw):
            owner_id = raw  # type: ignore[assignment]
            mint = False
        else:
            owner_id = _mint_owner_id()
            mint = True
        request.state.owner_id = owner_id
        response: Response = await call_next(request)
        if mint:
            response.set_cookie(
                self._cookie_name,
                owner_id,
                max_age=self._max_age,
                httponly=True,
                samesite="lax",
                secure=self._secure,
                path="/",
            )
        return response


def get_owner_id(request: Request) -> str:
    """FastAPI dependency that returns the request's resolved owner id.

    Args:
        request: The incoming request.

    Returns:
        str: The owner identifier attached by :class:`IdentityMiddleware`.

    Raises:
        RuntimeError: If the middleware is not active.
    """
    owner_id = getattr(request.state, "owner_id", None)
    if not isinstance(owner_id, str):
        raise RuntimeError(
            "owner_id is not on request.state — is IdentityMiddleware registered?"
        )
    return owner_id


__all__ = ["COOKIE_NAME", "IdentityMiddleware", "get_owner_id"]
