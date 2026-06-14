"""Request-principal resolution for the Nextext API.

The backend has no authentication. Identity is resolved per request from a
trusted header (default ``X-Auth-User``), with an optional environment-variable
fallback so developers — and any header-less client — resolve to a single
configured identity instead of being rejected:

1. If the configured trusted header is present, return its value.
2. Otherwise, if ``NEXTEXT_DEFAULT_IDENTITY`` is configured, return it (the
   dev / pre-auth fallback).
3. Otherwise fail closed with HTTP 401.

The Streamlit frontend carries a stable per-browser identifier in its URL
(``?owner=<uuid>``) and forwards it under the trusted header on every request,
so two browsers stay isolated for the purpose of scoping their in-memory jobs.

Trust model: the backend trusts anyone who can reach ``inference-net``. The
header just lets the backend tell callers apart. This module is the single
seam a real auth track would replace — swap the header read for a
verified-token read and nothing downstream (ownership checks, routes) changes.
"""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from nextext.utils.env_cfg import load_principal_env


def resolve_principal(request: Request) -> str:
    """FastAPI dependency that resolves the calling principal (owner id).

    Resolution order:

    1. If the configured trusted header is present and non-blank, return it.
    2. Otherwise, if a default identity is configured, return it.
    3. Otherwise fail closed with HTTP 401.

    Args:
        request: The incoming FastAPI/Starlette request.

    Returns:
        str: The resolved principal identifier used to scope owned jobs.

    Raises:
        HTTPException: 401 when neither the trusted header nor a configured
            default identity is available.
    """
    cfg = load_principal_env()
    header_value = request.headers.get(cfg.header_name)
    if header_value and header_value.strip():
        return header_value.strip()
    if cfg.default_identity:
        return cfg.default_identity
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing authenticated principal.",
    )


__all__ = ["resolve_principal"]
