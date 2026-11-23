"""Readiness of the transcription path, probed rather than inferred.

``GET /api/v1/health`` asks the central router for its model list, which the
router answers while a backend behind it (the Whisper one, say) is down. This
module asks the question that matters to a caller: does the Whisper endpoint
transcribe? It sends half a second of synthetic silence through the same
client, key and model the transcriber uses and counts any successful response
as yes, any exception (connection, status, configuration) as no.

The probe carries no user data, needs no identity and is not audited, like
``/health``; it never goes through :func:`nextext.core.inference_trace.record_inference`,
so no job's tally or audit event ever sees it.
"""

from __future__ import annotations

import asyncio
import io
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass

import openai
from loguru import logger

from nextext.core.transcription import build_whisper_client, whisper_calls_in_flight
from nextext.utils.env_cfg import load_whisper_env
from nextext.utils.log_cfg import error_label

#: How long a probe result, good or bad, answers every ``/ready`` request. A
#: caller that polls (Ghostwriter 0.6.1 and later) costs the Whisper endpoint at
#: most one request per period, however many callers there are.
READY_CACHE_TTL_S: float = 30.0

#: Seconds the probe waits for the Whisper endpoint, with no retries: a slow
#: or hung endpoint is not ready, and the answer must come back before the
#: callers' own timeouts.
PROBE_TIMEOUT_S: float = 10.0

PROBE_SECONDS: float = 0.5
PROBE_SAMPLE_RATE: int = 16_000


def silence_wav(seconds: float = PROBE_SECONDS, sample_rate: int = PROBE_SAMPLE_RATE) -> bytes:
    """Render silence as a mono 16-bit PCM WAV in memory.

    Args:
        seconds (float): Length of the clip.
        sample_rate (int): Samples per second.

    Returns:
        bytes: A complete WAV file of zero samples.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(2 * int(seconds * sample_rate)))
    return buffer.getvalue()


_PROBE_AUDIO: bytes = silence_wav()


class WhisperNotConfigured(Exception):
    """The Whisper settings cannot work (Ollama without a Whisper endpoint)."""


def probe_transcription() -> None:
    """Send half a second of synthetic silence to the Whisper endpoint.

    Uses :func:`nextext.utils.env_cfg.load_whisper_env` for the endpoint, key
    and model, as the transcriber does, with the cheapest response format
    (``json``), a :data:`PROBE_TIMEOUT_S` timeout and no retries.

    Raises:
        WhisperNotConfigured: The settings cannot work, which
            ``load_whisper_env`` reports as a ``RuntimeError``.
        Exception: Whatever the request raised — a connection error or an
            HTTP status error.
    """
    try:
        cfg = load_whisper_env()
    except RuntimeError as exc:
        raise WhisperNotConfigured from exc
    with build_whisper_client(cfg, timeout=PROBE_TIMEOUT_S, max_retries=0) as client:
        client.audio.transcriptions.create(
            model=cfg.model,
            file=("probe.wav", _PROBE_AUDIO, "audio/wav"),
            response_format="json",
        )


#: What the log says for a configuration that cannot work. Fixed text: the
#: probe never logs an exception's message.
NOT_CONFIGURED_HINT: str = (
    "Whisper is not configured (INFERENCE_PROVIDER=ollama needs WHISPER_API_BASE and WHISPER_MODEL)"
)


@dataclass(frozen=True)
class _ProbeResult:
    """What one probe found.

    Attributes:
        failure: ``None`` when Whisper transcribed, else what to log.
        busy: The probe timed out while this process had a Whisper request in
            flight, so the timeout may be the queue and not an outage.
    """

    failure: str | None
    busy: bool = False


class TranscriptionReadiness:
    """The cached, single-flight answer to "does Whisper transcribe?".

    A result holds for ``ttl_s`` seconds. The probe in flight is one
    :class:`asyncio.Task` that every waiting request awaits through
    :func:`asyncio.shield`, so requests that arrive during a probe share its
    result instead of starting their own, and a request that is cancelled
    (a disconnect, a shutdown) neither stops the probe nor lets the next
    request start a second one. The blocking probe runs on a worker thread, so
    the event loop stays free. A failure is logged once per change of state,
    by exception class and HTTP status only (never a response body), and the
    recovery once as well.

    A busy Whisper is not a down one. A probe that **times out** while this
    process has a Whisper request of its own in flight (a job's upload, which
    on a CPU backend can queue the probe behind minutes of audio) keeps the
    last good answer, and says so at debug. A timeout with nothing of ours in
    flight, or with no good answer yet, is not ready; a connection error, an
    HTTP error status or a configuration that cannot work is not ready
    whatever is in flight.

    :meth:`invalidate` marks the answer stale, so the next request probes
    afresh: a job that found inference down is evidence the cached ``200``
    is wrong.
    """

    def __init__(
        self,
        *,
        probe: Callable[[], None] = probe_transcription,
        clock: Callable[[], float] = time.monotonic,
        ttl_s: float = READY_CACHE_TTL_S,
        calls_in_flight: Callable[[], int] = whisper_calls_in_flight,
    ) -> None:
        """Set up an empty cache.

        Args:
            probe (Callable[[], None]): Raises when the endpoint does not
                transcribe; the real probe unless a test injects one.
            clock (Callable[[], float]): Monotonic seconds, injectable for tests.
            ttl_s (float): How long a result holds.
            calls_in_flight (Callable[[], int]): How many Whisper requests this
                process has in flight; the transcriber's own count unless a
                test injects one.
        """
        self._probe = probe
        self._clock = clock
        self._ttl_s = ttl_s
        self._calls_in_flight = calls_in_flight
        self._checked_at = 0.0
        self._ready: bool | None = None
        self._in_flight: asyncio.Task[bool] | None = None
        # Bumped by ``invalidate``; an answer counts only under the generation
        # it was stored in, which also stops a pre-invalidation good answer
        # from being kept while Whisper is busy.
        self._generation = 0
        self._stored_in = 0

    def invalidate(self) -> None:
        """Mark the answer stale: the next request probes afresh.

        A probe already in flight still answers the requests waiting on it, but
        its result is stored stale too, so the request after it probes again.
        """
        self._generation += 1

    async def is_ready(self) -> bool:
        """Report whether the Whisper endpoint transcribes, probing when stale.

        Returns:
            bool: ``True`` when the last probe, at most ``ttl_s`` seconds old,
                succeeded (or timed out while Whisper was busy with a job).
        """
        if (
            self._ready is not None
            and self._stored_in == self._generation
            and self._clock() - self._checked_at < self._ttl_s
        ):
            return self._ready
        if self._in_flight is None:
            self._in_flight = asyncio.create_task(self._probe_and_store())
        return await asyncio.shield(self._in_flight)

    async def _probe_and_store(self) -> bool:
        """Probe once, keep the result and clear the probe in flight.

        Returns:
            bool: Whether the endpoint transcribed (or was busy, after a good answer).
        """
        generation = self._generation
        try:
            result = await asyncio.to_thread(self._run_probe)
            failure = result.failure
            if result.busy and self._ready is True and self._stored_in == generation:
                logger.debug("Transcription readiness probe timed out while Whisper is busy; keeping the last answer.")
                failure = None
            self._checked_at = self._clock()
            self._note(failure)
            self._ready = failure is None
            self._stored_in = generation
            return self._ready
        finally:
            self._in_flight = None

    def _run_probe(self) -> _ProbeResult:
        """Run the probe once, on a worker thread.

        Returns:
            _ProbeResult: ``failure`` is ``None`` on success, else a fixed hint
                for a configuration that cannot work, otherwise the exception
                named by class and HTTP status; ``busy`` is true for a timeout
                while this process has a Whisper request in flight.
        """
        try:
            self._probe()
        except WhisperNotConfigured:
            return _ProbeResult(NOT_CONFIGURED_HINT)
        except openai.APITimeoutError as exc:
            return _ProbeResult(error_label(exc), busy=self._calls_in_flight() > 0)
        except Exception as exc:
            return _ProbeResult(error_label(exc))
        return _ProbeResult(None)

    def _note(self, failure: str | None) -> None:
        """Log a change of state: a failure once, and its recovery once.

        Args:
            failure (str | None): The probe's outcome, as :meth:`_run_probe` returned it.
        """
        if failure is not None:
            if self._ready is not False:
                logger.warning("Transcription readiness probe failed: {}", failure)
        elif self._ready is False:
            logger.info("Transcription readiness probe succeeded again.")
