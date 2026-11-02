"""Repo-wide test fixtures.

This file is intentionally small: most fixtures live close to the tests
that use them (see ``tests/test_api/conftest.py``). Anything here applies
to the whole suite.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from loguru import logger


@pytest.fixture(autouse=True)
def _hermetic_response_language_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear ``RESPONSE_LANGUAGE``/``NEXTEXT_RESPONSE_LANGUAGE`` before every test.

    Several modules (``nextext/utils/env_cfg.py``, ``nextext/core/openai_cfg.py``,
    ``nextext/core/words.py``, ``nextext/utils/model_loader.py``) call
    ``load_dotenv()`` at import time. ``load_dotenv()`` writes any value from a
    local, uncommitted ``.env`` straight into the real process environment (not
    a test-scoped sandbox) the first time one of those modules is imported
    during collection — and it sticks for the rest of the pytest session. A
    developer's ``.env`` commonly sets ``RESPONSE_LANGUAGE=de`` for local
    German-UI testing (see ``.env.example``), which then silently outranks the
    "unset" default every test in ``test_env_cfg.py``/``test_openai_cfg.py``
    assumes, making outcomes depend on whichever machine runs the suite rather
    than on the test itself.

    Delegating the delenv to ``monkeypatch`` here (rather than requiring each
    test to remember it) makes every test hermetic against ambient
    environment/`.env` state; tests that care about a specific value still set
    it explicitly via ``monkeypatch.setenv``.

    Args:
        monkeypatch: Fixture for patching environment variables.
    """
    monkeypatch.delenv("RESPONSE_LANGUAGE", raising=False)
    monkeypatch.delenv("NEXTEXT_RESPONSE_LANGUAGE", raising=False)


@pytest.fixture(autouse=True)
def _hermetic_identity_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the identity variables before every test.

    The API tests send the owner under the default ``X-Auth-User`` header and
    assume the header is trusted from any caller. A local ``.env`` (copied
    into the process environment at import time, see above) that sets an
    edge secret, service tokens, a default identity, or another header name
    would otherwise change how every request resolves; tests that need one
    set it explicitly.

    Args:
        monkeypatch: Fixture for patching environment variables.
    """
    for name in (
        "NEXTEXT_AUTH_HEADER",
        "NEXTEXT_DEFAULT_IDENTITY",
        "NEXTEXT_EDGE_SECRET",
        "NEXTEXT_SERVICE_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def log_lines() -> Iterator[list[str]]:
    """Capture every loguru message emitted while a test runs.

    Each entry is the rendered message plus any traceback loguru attaches, so
    a test can assert that content stays out of the operational logs. Request
    this fixture after any fixture that starts the app lifespan:
    ``setup_logging()`` removes every sink, this one included.

    Yields:
        list[str]: The captured messages, in emission order.
    """
    lines: list[str] = []
    sink_id = logger.add(lines.append, level="DEBUG", format="{message}")
    try:
        yield lines
    finally:
        logger.remove(sink_id)
