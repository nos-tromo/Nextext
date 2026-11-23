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

from loguru import logger

from nextext.core.transcription import build_whisper_client
from nextext.utils.env_cfg import load_whisper_env
from nextext.utils.log_cfg import error_label

#: How long a probe result, good or bad, answers every ``/ready`` request. A
#: caller that polls (Ghostwriter does) costs the Whisper endpoint at most one
#: request per period, however many callers there are.
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


def probe_transcription() -> None:
    """Send half a second of synthetic silence to the Whisper endpoint.

    Uses :func:`nextext.utils.env_cfg.load_whisper_env` for the endpoint, key
    and model, as the transcriber does, with the cheapest response format
    (``json``), a :data:`PROBE_TIMEOUT_S` timeout and no retries.

    Raises:
        Exception: Whatever the request raised — a connection error, an HTTP
            status error, or the ``RuntimeError`` of a configuration that
            cannot work (Ollama without ``WHISPER_API_BASE``).
    """
    cfg = load_whisper_env()
    with build_whisper_client(cfg, timeout=PROBE_TIMEOUT_S, max_retries=0) as client:
        client.audio.transcriptions.create(
            model=cfg.model,
            file=("probe.wav", _PROBE_AUDIO, "audio/wav"),
            response_format="json",
        )


class TranscriptionReadiness:
    """The cached, single-flight answer to "does Whisper transcribe?".

    A result holds for ``ttl_s`` seconds. Requests that arrive while a probe is
    running wait for it and share its result instead of starting their own, and
    the blocking probe runs on a worker thread, so the event loop stays free.
    A failure is logged once per change of state, by exception class and HTTP
    status only (never a response body), and the recovery once as well.
    """

    def __init__(
        self,
        *,
        probe: Callable[[], None] = probe_transcription,
        clock: Callable[[], float] = time.monotonic,
        ttl_s: float = READY_CACHE_TTL_S,
    ) -> None:
        """Set up an empty cache.

        Args:
            probe (Callable[[], None]): Raises when the endpoint does not
                transcribe; the real probe unless a test injects one.
            clock (Callable[[], float]): Monotonic seconds, injectable for tests.
            ttl_s (float): How long a result holds.
        """
        self._probe = probe
        self._clock = clock
        self._ttl_s = ttl_s
        self._lock = asyncio.Lock()
        self._checked_at = 0.0
        self._ready: bool | None = None

    async def is_ready(self) -> bool:
        """Report whether the Whisper endpoint transcribes, probing when stale.

        Returns:
            bool: ``True`` when the last probe, at most ``ttl_s`` seconds old,
                succeeded.
        """
        async with self._lock:
            if self._ready is not None and self._clock() - self._checked_at < self._ttl_s:
                return self._ready
            failure = await asyncio.to_thread(self._run_probe)
            self._checked_at = self._clock()
            self._note(failure)
            self._ready = failure is None
            return self._ready

    def _run_probe(self) -> str | None:
        """Run the probe once, on a worker thread.

        Returns:
            str | None: ``None`` on success, else the failure named by class
                and HTTP status.
        """
        try:
            self._probe()
        except Exception as exc:
            return error_label(exc)
        return None

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
