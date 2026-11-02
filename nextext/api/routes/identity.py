"""Signed-in identity route (``GET /whoami``), for the SPA's AppHeader."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from nextext.api.identity import identity_headers_trusted, resolve_principal
from nextext.api.schemas import WhoamiResponse

router = APIRouter(tags=["identity"])


@router.get("/whoami", response_model=WhoamiResponse)
def get_whoami(
    request: Request,
    principal: str = Depends(resolve_principal),
) -> WhoamiResponse:
    """Return the resolved calling identity, for the SPA's AppHeader.

    Principal-gated like every job-scoped endpoint (401 without a trusted
    header or a configured dev default identity) — unlike ``/health``,
    ``/version``, and ``/config``, which are deliberately unauthenticated.

    ``display_name`` is read straight off the ``X-Auth-Name`` request header
    (Authelia's displayname, injected by the edge gateway) and is purely
    decorative — it plays no part in identity/principal resolution, unlike
    ``username``. ``None`` when the header is absent (dev without the
    gateway in front) or, with ``NEXTEXT_EDGE_SECRET`` set, when the request
    is not edge-verified — the gateway's headers count only from the gateway.

    Args:
        request: The incoming request, for the decorative ``X-Auth-Name``
            header.
        principal: The resolved request principal (same dependency used by
            every other authenticated route).

    Returns:
        WhoamiResponse: The caller's resolved principal name plus, if
            present and trusted, the gateway's decorative display name.
    """
    display_name = request.headers.get("X-Auth-Name") if identity_headers_trusted(request) else None
    return WhoamiResponse(username=principal, display_name=display_name)


__all__ = ["router"]
