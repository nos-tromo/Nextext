"""Repo-wide test fixtures.

This file is intentionally small: most fixtures live close to the tests
that use them (see ``tests/test_api/conftest.py``). Anything here applies
to the whole suite.
"""

from __future__ import annotations

import pytest


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
