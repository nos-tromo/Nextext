"""Regression tests for the frontend nginx proxy config.

Guards the ``proxy_read_timeout`` budget for on-demand artifact downloads.
The backend assembles CSV/XLSX/ZIP bundles in memory on request; a multi-file
batch ZIP can take far longer than nginx's default 60s to build, so the
download locations must carry an explicit long read timeout. Without it the
request falls through to ``location /api/`` (no timeout → 60s default) and the
browser gets a ``504 Gateway Time-out`` mid-build.
"""

from __future__ import annotations

import re
from pathlib import Path

_CONF = Path(__file__).resolve().parents[1] / "frontend" / "nginx" / "default.conf"

# Representative download request paths that must each resolve to a location
# with a generous read timeout rather than the generic ``/api/`` fallthrough.
_DOWNLOAD_PATHS = (
    "/api/v1/jobs/9d1f7c2e-0000-4a00-8000-000000000000/artifacts/archive.zip",
    "/api/v1/jobs/batch/archive.zip",
)

_UNIT_SECONDS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}


def _regex_locations(conf: str) -> list[tuple[str, str]]:
    """Extract ``location ~ <regex> { <body> }`` blocks in file order.

    Args:
        conf: Full nginx config text.

    Returns:
        list[tuple[str, str]]: ``(regex, body)`` per regex location, ordered as
            nginx evaluates them (first match wins).
    """
    blocks: list[tuple[str, str]] = []
    for match in re.finditer(r"location\s+~\s+(\S+)\s*\{", conf):
        body_start = match.end()
        body_end = conf.index("}", body_start)
        blocks.append((match.group(1), conf[body_start:body_end]))
    return blocks


def _read_timeout_seconds(body: str) -> int | None:
    """Return a location body's ``proxy_read_timeout`` in seconds.

    Args:
        body: Text between a location block's braces.

    Returns:
        int | None: The timeout in seconds, or ``None`` when unset.
    """
    match = re.search(r"proxy_read_timeout\s+(\d+)([smhd]?)", body)
    if match is None:
        return None
    return int(match.group(1)) * _UNIT_SECONDS[match.group(2)]


def test_download_locations_have_long_read_timeout() -> None:
    """Artifact + batch downloads resolve to a generous ``proxy_read_timeout``.

    For each representative download path, the first regex location that
    matches it (nginx first-match-wins) must declare an explicit read timeout
    well above the 60s default; otherwise the path falls through to the generic
    ``location /api/`` block and a slow multi-file batch ZIP yields a 504.
    """
    conf = _CONF.read_text(encoding="utf-8")
    locations = _regex_locations(conf)

    for path in _DOWNLOAD_PATHS:
        winner = next(((rx, body) for rx, body in locations if re.match(rx, path)), None)
        assert winner is not None, (
            f"No regex location matches {path!r}; it falls through to "
            "`location /api/` (60s default) and large downloads will 504."
        )
        seconds = _read_timeout_seconds(winner[1])
        assert seconds is not None and seconds >= 300, (
            f"{path!r} resolves to `location ~ {winner[0]}` with "
            f"proxy_read_timeout={seconds!r}s; expected an explicit budget >= 300s."
        )
