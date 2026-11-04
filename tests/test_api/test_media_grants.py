"""Tests for the in-memory media playback grants (``nextext/api/media_grants.py``)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from nextext.api.media_grants import MAX_GRANTS_PER_JOB, MediaGrants
from nextext.audit import Origin
from nextext.utils.audit_event import Actor, Purpose

ORIGIN = Origin(
    actor=Actor("alice.example", "edge"),
    purpose=Purpose(case_ref="case-0001", reason="Review of a reported clip"),
    request_id="7c9e6679-7425-40de-944b-e07fc1f90ae7",
)
NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


class _Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self) -> None:
        """Start at an arbitrary instant."""
        self.now = 1000.0

    def __call__(self) -> float:
        """Return the current time.

        Returns:
            float: Seconds.
        """
        return self.now


def _store(ttl_s: int = 900) -> tuple[MediaGrants, _Clock]:
    """Build a store on a hand-moved clock.

    Args:
        ttl_s: The grant lifetime.

    Returns:
        tuple[MediaGrants, _Clock]: The store and its clock.
    """
    clock = _Clock()
    return MediaGrants(ttl_s, clock=clock, wall_clock=lambda: NOW), clock


def test_a_grant_binds_its_job_and_origin() -> None:
    """The token resolves to the grant for its job, carrying who asked, why and under which request."""
    store, _ = _store()
    token, grant = store.issue("job-1", ORIGIN)

    assert len(token) >= 40
    assert store.resolve("job-1", token) == grant
    assert grant.job_id == "job-1"
    assert grant.origin == ORIGIN
    assert grant.expires_at == NOW + timedelta(seconds=900)


def test_the_grant_id_is_not_the_token_and_the_token_is_not_kept() -> None:
    """The audit log records the grant id, which reveals nothing of the token; the store keeps only a hash."""
    store, _ = _store()
    token, grant = store.issue("job-1", ORIGIN)

    assert grant.grant_id
    assert grant.grant_id not in token
    assert token not in repr(store._grants)


def test_a_grant_opens_only_its_own_job() -> None:
    """A grant for one job is no key to another, and an unknown token opens nothing."""
    store, _ = _store()
    token, _ = store.issue("job-1", ORIGIN)

    assert store.resolve("job-2", token) is None
    assert store.resolve("job-1", "not-a-grant") is None


def test_a_grant_expires_after_its_lifetime() -> None:
    """Usable for every range of a playback until the lifetime is over, then gone."""
    store, clock = _store(ttl_s=60)
    token, grant = store.issue("job-1", ORIGIN)

    clock.now += 59
    assert store.resolve("job-1", token) == grant
    clock.now += 1
    assert store.resolve("job-1", token) is None
    assert len(store) == 0


def test_issuing_drops_expired_grants() -> None:
    """Expired grants do not pile up in memory."""
    store, clock = _store(ttl_s=60)
    store.issue("job-1", ORIGIN)
    clock.now += 61
    store.issue("job-2", ORIGIN)

    assert len(store) == 1


def test_a_job_keeps_a_bounded_number_of_grants() -> None:
    """Asking in a loop cannot grow memory without bound: the oldest grant of the job makes room."""
    store, _ = _store()
    tokens = [store.issue("job-1", ORIGIN)[0] for _ in range(MAX_GRANTS_PER_JOB + 1)]
    other, _ = store.issue("job-2", ORIGIN)

    assert store.resolve("job-1", tokens[0]) is None
    assert all(store.resolve("job-1", token) is not None for token in tokens[1:])
    assert store.resolve("job-2", other) is not None


def test_revoking_a_job_drops_only_its_grants() -> None:
    """A deleted job's grants stop working; other jobs' keep theirs."""
    store, _ = _store()
    first, _ = store.issue("job-1", ORIGIN)
    second, _ = store.issue("job-2", ORIGIN)

    store.revoke_job("job-1")

    assert store.resolve("job-1", first) is None
    assert store.resolve("job-2", second) is not None
    store.clear()
    assert len(store) == 0
