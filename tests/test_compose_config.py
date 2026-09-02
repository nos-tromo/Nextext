"""Regression tests for the Docker Compose configuration.

Pins the cache-directory env vars the read-only-rootfs hardening (deploy
ADR 0001) depends on. The backend container runs as uid 10001 with a
read-only root filesystem, so every library that writes a cache at import
time must be pointed at one of the writable mounts
(`/home/app/.cache/spacy`, `/tmp`). matplotlib was the one missed in the
original sweep: without ``MPLCONFIGDIR`` it tries
``$HOME/.config/matplotlib``, fails with ``[Errno 30] Read-only file
system`` at app import, and falls back to a throwaway per-process
``/tmp/matplotlib-*`` dir (log noise plus a font-cache rebuild every boot).
"""

from __future__ import annotations

import re
from pathlib import Path

_COMPOSE = Path(__file__).resolve().parents[1] / "docker" / "compose.yaml"

# Writable mountpoints inside the backend container (the compose volume list).
_WRITABLE_ROOTS = ("/home/app/.cache/spacy", "/tmp")

# Env var -> exact path the backend service must set it to.
_CACHE_ENV_VARS = {
    "SPACY_MODEL_DIR": "/home/app/.cache/spacy",
    "MPLCONFIGDIR": "/tmp/matplotlib",
}


def _env_value(compose: str, name: str) -> str | None:
    """Return the value a ``NAME: value`` mapping line assigns in the compose file.

    Args:
        compose: Full compose.yaml text.
        name: Environment variable name to look up.

    Returns:
        str | None: The assigned value, or ``None`` when the variable is not set.
    """
    match = re.search(rf"^\s+{re.escape(name)}:\s*(\S+)\s*$", compose, re.MULTILINE)
    if match is None:
        return None
    return match.group(1)


def test_backend_cache_env_vars_point_at_writable_mounts() -> None:
    """Each import-time cache dir is pinned to a path under a writable mount.

    The read-only rootfs (deploy ADR 0001) makes ``$HOME``-derived cache
    defaults fail, so the compose file must set each cache env var explicitly
    and the value must live under one of the backend's writable volumes.
    """
    compose = _COMPOSE.read_text(encoding="utf-8")

    for name, expected in _CACHE_ENV_VARS.items():
        value = _env_value(compose, name)
        assert value == expected, (
            f"compose.yaml must set {name}: {expected} on the backend service "
            f"(found {value!r}); without it the read-only rootfs breaks the "
            "library's $HOME-derived cache default."
        )
        assert any(expected == root or expected.startswith(root + "/") for root in _WRITABLE_ROOTS), (
            f"{name}={expected!r} is not under a writable mount {_WRITABLE_ROOTS}."
        )
