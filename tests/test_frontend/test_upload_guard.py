"""Tests for the frontend upload-size guard.

Streamlit's ``file_uploader`` holds the whole selection in the Streamlit
server's memory, so a multi-GB batch can exhaust RAM before any pipeline work
starts. ``check_batch_within_limit`` turns that silent crash into an
actionable message, bounded by the ``NEXTEXT_MAX_BATCH_MB`` env knob.
"""

from __future__ import annotations

import pytest

from nextext.frontend.app import _max_batch_bytes
from nextext.frontend.state import check_batch_within_limit


class _FakeUpload:
    """A minimal stand-in for a Streamlit ``UploadedFile`` (only ``size``)."""

    def __init__(self, size: int) -> None:
        """Store the reported byte size.

        Args:
            size: File size in bytes.
        """
        self.size = size


def test_check_batch_within_limit_allows_batch_under_cap() -> None:
    """A batch whose total is at or under the cap is accepted (returns None)."""
    files = [_FakeUpload(40), _FakeUpload(60)]
    assert check_batch_within_limit(files, max_total_bytes=100) is None


def test_check_batch_within_limit_flags_batch_over_cap() -> None:
    """An oversized batch returns an actionable message naming the CLI escape."""
    files = [_FakeUpload(60), _FakeUpload(60)]

    message = check_batch_within_limit(files, max_total_bytes=100)

    assert message is not None
    assert "nextext-cli" in message


def test_check_batch_within_limit_allows_empty_selection() -> None:
    """No files selected is never over the limit."""
    assert check_batch_within_limit([], max_total_bytes=100) is None


def test_check_batch_within_limit_tolerates_missing_size() -> None:
    """Files lacking a usable ``size`` are treated as zero bytes, not crashes."""
    file_without_size = object()
    assert check_batch_within_limit([file_without_size], max_total_bytes=100) is None


def test_max_batch_bytes_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cap defaults to 2048 MiB when the env var is unset."""
    monkeypatch.delenv("NEXTEXT_MAX_BATCH_MB", raising=False)
    assert _max_batch_bytes() == 2048 * (1 << 20)


def test_max_batch_bytes_honours_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """A numeric override is read from ``NEXTEXT_MAX_BATCH_MB``."""
    monkeypatch.setenv("NEXTEXT_MAX_BATCH_MB", "512")
    assert _max_batch_bytes() == 512 * (1 << 20)


def test_max_batch_bytes_falls_back_on_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-integer override falls back to the default rather than raising."""
    monkeypatch.setenv("NEXTEXT_MAX_BATCH_MB", "not-a-number")
    assert _max_batch_bytes() == 2048 * (1 << 20)
