"""Media playback grants: a short-lived capability for one playback.

A ``<video>``/``<audio>`` element cannot send headers — neither the trusted
identity header nor ``X-Audit-Purpose`` — so the media route cannot learn
from the request itself why a recording is played. The SPA therefore asks
for a grant first (``POST /jobs/{id}/media-grant``, an ordinary request that
states the purpose) and plays the URL it gets back. The grant binds, on the
server, a fresh random token to the job and to the origin of that request:
the caller, the stated purpose and the request id. Each range request the
player makes presents the token until it expires, and its ``media.stream``
event takes the purpose from the grant (:mod:`nextext.api.routes.jobs`).

Grants live in memory like the jobs they open and die with the process; a
job's grants are revoked when the job is deleted. Only a SHA-256 of each
token is kept, so a lookup cannot be timed against a stored token's prefix.
Each grant also carries a separate random id, which the audit log records to
tie the ranges of one playback together — never the token.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from nextext.audit import Origin

__all__ = ["MAX_GRANTS_PER_JOB", "MediaGrant", "MediaGrants"]

#: Live grants kept per job; issuing one more drops the oldest. A playback
#: needs one, so this only bounds the memory a client asking in a loop costs.
MAX_GRANTS_PER_JOB = 16


@dataclass(frozen=True)
class MediaGrant:
    """One grant, as the server keeps it.

    Attributes:
        grant_id: Random id the audit log records, unrelated to the token.
        job_id: The job whose upload the grant plays.
        origin: The caller, stated purpose and request id of the request the
            grant was issued to.
        expires_at: When it stops working, on the wall clock (for the client).
        expires_mono: The same moment on the monotonic clock (for the checks).
    """

    grant_id: str
    job_id: str
    origin: Origin
    expires_at: datetime
    expires_mono: float


def _key(token: str) -> str:
    """Return the key a token is kept under: its SHA-256, never the token.

    Args:
        token: A grant token, as issued or as presented.

    Returns:
        str: The hex digest.
    """
    return hashlib.sha256(token.encode()).hexdigest()


class MediaGrants:
    """The live media grants of one process.

    Every method runs on the event loop and never awaits, so no lock is
    needed.
    """

    def __init__(
        self,
        ttl_s: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
    ) -> None:
        """Create an empty store.

        Args:
            ttl_s: Seconds a grant stays usable.
            clock: Monotonic time source for expiry checks.
            wall_clock: Wall-clock source for the expiry reported to clients.
        """
        self.ttl_s = ttl_s
        self._clock = clock
        self._wall_clock = wall_clock
        self._grants: dict[str, MediaGrant] = {}

    def __len__(self) -> int:
        """Return how many grants are held, expired ones not yet dropped included.

        Returns:
            int: The count.
        """
        return len(self._grants)

    def _prune(self, now: float) -> None:
        """Drop every expired grant.

        Args:
            now: The current monotonic time.
        """
        for key in [key for key, grant in self._grants.items() if grant.expires_mono <= now]:
            del self._grants[key]

    def issue(self, job_id: str, origin: Origin) -> tuple[str, MediaGrant]:
        """Issue a grant for one job to the request described by ``origin``.

        Args:
            job_id: The job whose upload it plays.
            origin: The requesting caller, its stated purpose and request id.

        Returns:
            tuple[str, MediaGrant]: The token (handed to the client once and
                not kept) and the grant.
        """
        now = self._clock()
        self._prune(now)
        # Insertion order is issue order, and every grant lives as long, so the
        # first ones found are the oldest.
        same_job = [key for key, grant in self._grants.items() if grant.job_id == job_id]
        for key in same_job[: max(0, len(same_job) - MAX_GRANTS_PER_JOB + 1)]:
            del self._grants[key]
        token = secrets.token_urlsafe(32)
        grant = MediaGrant(
            grant_id=secrets.token_hex(8),
            job_id=job_id,
            origin=origin,
            expires_at=self._wall_clock() + timedelta(seconds=self.ttl_s),
            expires_mono=now + self.ttl_s,
        )
        self._grants[_key(token)] = grant
        return token, grant

    def resolve(self, job_id: str, token: str) -> MediaGrant | None:
        """Return the live grant a token names for a job.

        Args:
            job_id: The job named by the request path.
            token: The presented token.

        Returns:
            MediaGrant | None: The grant, or ``None`` when the token is
                unknown, expired, or was issued for another job.
        """
        key = _key(token)
        grant = self._grants.get(key)
        if grant is None:
            return None
        if grant.expires_mono <= self._clock():
            del self._grants[key]
            return None
        return grant if grant.job_id == job_id else None

    def revoke_job(self, job_id: str) -> None:
        """Drop every grant of a job, e.g. once it is deleted.

        Args:
            job_id: The job.
        """
        for key in [key for key, grant in self._grants.items() if grant.job_id == job_id]:
            del self._grants[key]

    def clear(self) -> None:
        """Drop every grant."""
        self._grants.clear()
