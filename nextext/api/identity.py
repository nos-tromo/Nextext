"""Header-based owner identity for the Nextext API.

The backend has no authentication. Identity is anonymous and lives in
the browser's ``localStorage`` (managed by the Streamlit frontend). The
browser sends a stable UUID4 hex in the ``X-Owner-Id`` header on every
request; the backend uses that value to scope persistent rows so a
browser sees only its own saved jobs.

Why not a cookie? A cookie set by the FastAPI backend would attach to
the *Streamlit server's* ``httpx.Client`` jar — not to the user's
browser — because the backend is internal-only and the browser never
talks to it directly. Cookies in that arrangement are process-wide,
shared across browsers, and lost on backend redeploys. A header
sourced from the browser's own ``localStorage`` survives tab close,
browser restart, and backend redeploys, and isolates each browser from
every other.

Trust model is unchanged: the backend trusts anyone who can reach
``inference-net``. The header just lets the backend tell two browsers
apart for the purpose of filtering rows.
"""

from __future__ import annotations

import re
import uuid

from fastapi import Header, HTTPException, status

OWNER_HEADER = "X-Owner-Id"
_UUID4_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def _is_valid_owner_id(value: str | None) -> bool:
    """Return whether ``value`` is a 32-char hex string that parses as a UUID4.

    A strict parser rejects malformed values so the database never
    records a non-UUID owner identifier — which would break the
    rehydration path that reconstructs per-job filesystem paths from
    ``job_id`` and reuses the owner column as the privacy boundary.

    Args:
        value: Raw header value (or ``None``).

    Returns:
        bool: ``True`` only when ``value`` is exactly 32 hex characters
            that parse as a UUID.
    """
    if not value or not _UUID4_PATTERN.match(value):
        return False
    try:
        uuid.UUID(hex=value)
    except (TypeError, ValueError):
        return False
    return True


def get_owner_id(
    x_owner_id: str | None = Header(default=None, alias=OWNER_HEADER),
) -> str:
    """FastAPI dependency that returns the caller's owner identifier.

    Args:
        x_owner_id: Value of the ``X-Owner-Id`` request header.

    Returns:
        str: The validated 32-char hex UUID.

    Raises:
        HTTPException: 400 when the header is missing or malformed.
    """
    if not _is_valid_owner_id(x_owner_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Missing or invalid {OWNER_HEADER} header; "
                "expected a 32-character hex UUID."
            ),
        )
    assert x_owner_id is not None  # Narrowed by _is_valid_owner_id above.
    return x_owner_id


__all__ = ["OWNER_HEADER", "get_owner_id"]
