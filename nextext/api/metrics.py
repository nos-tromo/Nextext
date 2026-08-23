"""Job-outcome counters exposed on ``GET /metrics``.

The HTTP request/latency series come from ``prometheus_fastapi_instrumentator``
(wired in :mod:`nextext.api.main`); it cannot see *why* a job ended, so a job
that transcribed nothing is indistinguishable from one that transcribed an
hour of speech. These counters close that gap: an operator can alert on a rise
in skipped jobs (a broken VAD endpoint over-reporting silence) or in decode
failures (users uploading an unsupported container) without reading logs.

Labels are typed codes from :mod:`nextext.core.outcomes` only — never a
filename, owner id, or any transcript text. Counters register on the default
``prometheus_client`` registry, which is the one the instrumentator exposes.
"""

from __future__ import annotations

from prometheus_client import Counter

from nextext.core.outcomes import FailureCode, SkipReason

JOBS_TOTAL: Counter = Counter(
    "nextext_jobs_total",
    "Jobs by terminal outcome.",
    labelnames=("outcome",),
)

JOBS_SKIPPED_TOTAL: Counter = Counter(
    "nextext_jobs_skipped_total",
    "Jobs that completed without a transcript, by typed reason.",
    labelnames=("reason",),
)

JOBS_FAILED_TOTAL: Counter = Counter(
    "nextext_jobs_failed_total",
    "Jobs that failed, by typed failure code.",
    labelnames=("code",),
)


def record_completed() -> None:
    """Count a job that finished with a transcript."""
    JOBS_TOTAL.labels(outcome="completed").inc()


def record_skipped(reason: SkipReason | None) -> None:
    """Count a job that finished without a transcript.

    Args:
        reason (SkipReason | None): Typed cause; ``None`` is recorded as
            ``"unknown"`` so the series stays complete.
    """
    JOBS_TOTAL.labels(outcome="skipped").inc()
    JOBS_SKIPPED_TOTAL.labels(reason=reason or "unknown").inc()


def record_failed(code: FailureCode) -> None:
    """Count a failed job.

    Args:
        code (FailureCode): Typed failure cause.
    """
    JOBS_TOTAL.labels(outcome="failed").inc()
    JOBS_FAILED_TOTAL.labels(code=code).inc()
