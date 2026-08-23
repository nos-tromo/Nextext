"""Typed outcome vocabulary shared by the pipeline, the API, and the CLI.

A run can end without a transcript for three distinct reasons, and a job can
fail for reasons the user can act on (an undecodable upload) or cannot (an
inference outage). Both used to collapse into one string — a single hardcoded
``"No speech detected in audio file."`` and a generic ``"Job failed."`` — so
neither the user nor the logs could tell the cases apart.

The codes here are the machine-readable contract: the backend puts a code on
the job snapshot, SSE events, and its log lines and metrics; the frontend
localizes it. The human-readable English strings below are a fallback for
non-UI consumers (CLI logs, older clients) — the SPA renders its own
translations keyed off the code and never displays this prose.
"""

from typing import Literal, get_args

#: Why a completed job produced no transcript.
SkipReason = Literal[
    "vad_no_speech",
    "asr_empty_transcript",
    "asr_all_segments_filtered",
]

#: Why a job failed. ``internal`` covers everything the user cannot act on
#: (inference outage, bug); its detail stays in the logs, never in a response.
FailureCode = Literal[
    "undecodable_media",
    "internal",
]

SKIP_REASONS: tuple[str, ...] = get_args(SkipReason)
FAILURE_CODES: tuple[str, ...] = get_args(FailureCode)

_SKIP_REASON_TEXT: dict[str, str] = {
    "vad_no_speech": "No speech detected in the audio.",
    "asr_empty_transcript": "Transcription returned no text for this file.",
    "asr_all_segments_filtered": "Only non-speech audio was detected; all transcribed segments were discarded.",
}


def skip_reason_text(code: SkipReason | None) -> str:
    """Return the English fallback prose for a skip reason code.

    Args:
        code (SkipReason | None): The typed reason, or ``None`` when the
            cause is unknown (e.g. an empty transcript from a path that
            records no code).

    Returns:
        str: A human-readable sentence. Unknown or missing codes fall back
        to the generic "no processable speech" wording.
    """
    if code is None:
        return "No processable speech was found in the file."
    return _SKIP_REASON_TEXT.get(code, "No processable speech was found in the file.")
