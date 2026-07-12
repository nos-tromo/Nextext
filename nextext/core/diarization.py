"""Speaker-diarization agent: HTTP client for the out-of-process ``/diarize`` service.

Nextext no longer hosts pyannote in-process. Diarization runs against an HTTP
``/diarize`` endpoint (e.g. ``nos-tromo/vllm-service``) that accepts a media
upload and returns chronological speaker turns. This module owns both the wire
call and the client-side alignment of those turns onto Whisper's transcript
segments (by maximum temporal overlap), keeping the contract in one place.

The endpoint is resolved by :func:`nextext.utils.env_cfg.load_diarization_env`:
``DIARIZE_API_BASE`` when set, else the central ``OPENAI_API_BASE`` (one trailing
``/v1`` stripped). When neither resolves diarization is disabled and callers
simply receive no speaker labels. Failures are logged and swallowed: a transcript
without speakers is preferable to a failed job.
"""

from pathlib import Path
from typing import Any

import httpx as httpx  # explicit re-export so tests can monkeypatch diarization.httpx
from loguru import logger

from nextext.utils.env_cfg import load_diarization_env

__all__ = [
    "SPEAKER_LABEL_PREFIX",
    "assign_speakers_by_overlap",
    "build_speaker_segments",
    "canonicalize_speaker_labels",
    "diarize_file",
    "gate_turns_by_vad",
    "renumber_speakers_by_appearance",
]

SPEAKER_LABEL_PREFIX = "Speaker"


def canonicalize_speaker_labels(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Renumber raw diarization labels to contiguous ``Speaker N`` by first appearance.

    pyannote's ``SPEAKER_00``/``SPEAKER_02`` labels are arbitrary and gap-y.
    This maps them to ``Speaker 1``, ``Speaker 2``, … in the order each label
    is first heard (earliest turn ``start``), so the first voice is always
    ``Speaker 1``. The input order is preserved in the output; only the
    ``speaker`` string changes.

    Args:
        turns (list[dict[str, Any]]): Speaker turns with ``start`` / ``end`` /
            ``speaker`` keys, as returned by :func:`diarize_file`.

    Returns:
        list[dict[str, Any]]: New turn dicts (same order) with canonical labels.
    """
    mapping: dict[str, str] = {}
    for turn in sorted(turns, key=lambda t: float(t["start"])):
        raw = str(turn["speaker"])
        if raw not in mapping:
            mapping[raw] = f"{SPEAKER_LABEL_PREFIX} {len(mapping) + 1}"
    return [{**turn, "speaker": mapping[str(turn["speaker"])]} for turn in turns]


def renumber_speakers_by_appearance(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Renumber speaker labels to ``Speaker N`` by first appearance in the transcript.

    :func:`canonicalize_speaker_labels` numbers speakers by their earliest *turn*,
    but word-level alignment, VAD-gating, and sentence restoration can make a
    speaker first surface in the *assembled transcript* in a different order — so a
    reader sees e.g. ``Speaker 4`` before ``Speaker 1``. This final pass fixes the
    display order: walking the segments top to bottom, the first labelled speaker
    becomes ``Speaker 1``, the next new one ``Speaker 2``, and so on. It is the
    authoritative numbering the transcript renders.

    Segments without a ``speaker`` key (or an empty one) pass through untouched and
    do not consume a number. Labels are treated as opaque strings, so raw
    ``SPEAKER_xx`` and already-canonical ``Speaker N`` input are both accepted.

    Args:
        segments (list[dict[str, Any]]): The assembled transcript segments, in
            reading order, each optionally carrying a ``speaker`` key.

    Returns:
        list[dict[str, Any]]: New segment dicts (same order, other keys
            preserved) with speaker labels renumbered by first appearance.
    """
    mapping: dict[str, str] = {}
    renumbered: list[dict[str, Any]] = []
    for segment in segments:
        speaker = segment.get("speaker")
        if not speaker:
            renumbered.append(dict(segment))
            continue
        if speaker not in mapping:
            mapping[speaker] = f"{SPEAKER_LABEL_PREFIX} {len(mapping) + 1}"
        renumbered.append({**segment, "speaker": mapping[speaker]})
    return renumbered


def gate_turns_by_vad(
    turns: list[dict[str, Any]],
    vad_intervals: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    """Crop diarization turns to the VAD speech timeline.

    Intersects each turn's ``[start, end]`` with every speech interval, emitting
    one turn per overlapping piece: a turn spanning a non-speech gap (e.g. music
    between utterances) splits into its speech-only fragments, and a turn
    overlapping no speech is dropped. Speaker labels (and any other keys) are
    preserved. This suppresses the false alarm from pyannote over-detecting
    music/noise as speech (the "music scored as a speaker" defect).

    Empty ``vad_intervals`` returns ``turns`` unchanged — a fail-safe so an empty
    VAD result never blanks an otherwise-speech transcript.

    Args:
        turns (list[dict[str, Any]]): Diarization turns with float ``start`` /
            ``end`` and a ``speaker`` key, as returned by :func:`diarize_file`.
        vad_intervals (list[tuple[float, float]]): Chronological speech
            ``(start, end)`` intervals from the ``/vad`` service (see
            :func:`nextext.core.vad.speech_segments`).

    Returns:
        list[dict[str, Any]]: New turn dicts cropped to the speech intervals.
    """
    if not vad_intervals:
        return turns
    gated: list[dict[str, Any]] = []
    for turn in turns:
        start = float(turn["start"])
        end = float(turn["end"])
        for speech_start, speech_end in vad_intervals:
            overlap_start = max(start, speech_start)
            overlap_end = min(end, speech_end)
            if overlap_end > overlap_start:
                gated.append({**turn, "start": overlap_start, "end": overlap_end})
    return gated


def diarize_file(
    file_path: Path,
    *,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[dict[str, Any]]:
    """Request speaker turns for an audio file from the ``/diarize`` service.

    The service URL is ``{base}/diarize`` for the resolved diarization ``base``
    (``DIARIZE_API_BASE`` or the central ``OPENAI_API_BASE`` with one trailing
    ``/v1`` stripped). When neither resolves diarization is disabled: a warning is
    logged and an empty list is returned so the caller proceeds without speaker
    labels. Any transport or HTTP error is likewise logged and swallowed into an
    empty list.

    ``num_speakers`` (exact count) is mutually exclusive with
    ``min_speakers``/``max_speakers`` on the server side. The Nextext pipeline
    calls this with no bounds so pyannote auto-detects the speaker count; the
    bound parameters are retained for API completeness.

    Args:
        file_path (Path): Path to the audio/video file to diarize. Sent as-is;
            the server resamples to 16 kHz mono via ffmpeg.
        num_speakers (int | None): Exact number of speakers, if known.
        min_speakers (int | None): Lower bound on the speaker count.
        max_speakers (int | None): Upper bound on the speaker count.

    Returns:
        list[dict[str, Any]]: Chronological speaker turns, each a mapping with
            ``start`` / ``end`` (absolute seconds) and ``speaker`` keys. Empty
            when diarization is disabled or the request fails.
    """
    config = load_diarization_env()
    if not config.api_base:
        logger.warning(
            "Diarization requested but no endpoint is configured (DIARIZE_API_BASE "
            "and OPENAI_API_BASE both unset); returning no speaker turns. Set "
            "DIARIZE_API_BASE or the central OPENAI_API_BASE to enable speaker labels."
        )
        return []

    data: dict[str, int] = {}
    if num_speakers is not None:
        data["num_speakers"] = num_speakers
    if min_speakers is not None:
        data["min_speakers"] = min_speakers
    if max_speakers is not None:
        data["max_speakers"] = max_speakers

    headers: dict[str, str] = {}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    url = f"{config.api_base}/diarize"
    try:
        with open(file_path, "rb") as audio:
            response = httpx.post(
                url,
                files={"file": (file_path.name, audio, "application/octet-stream")},
                data=data,
                headers=headers,
                timeout=config.timeout,
            )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Diarization request to {} failed ({}): {}",
            url,
            exc.response.status_code,
            exc.response.text[:500],
        )
        return []
    except (httpx.HTTPError, ValueError, OSError) as exc:
        logger.error("Diarization request to {} failed: {}", url, exc)
        return []

    if not isinstance(payload, dict):
        logger.error("Diarization response from {} was not a JSON object; ignoring.", url)
        return []

    segments = list(payload.get("segments", []))
    logger.info("Diarization complete: {} speaker turns from '{}'.", len(segments), file_path.name)
    return segments


def _speaker_by_overlap(
    start: float,
    end: float,
    turns: list[dict[str, Any]],
) -> str | None:
    """Return the speaker with the greatest cumulative overlap of ``[start, end]``.

    Args:
        start (float): Window start in seconds.
        end (float): Window end in seconds.
        turns (list[dict[str, Any]]): Speaker turns with ``start`` / ``end`` /
            ``speaker`` keys.

    Returns:
        str | None: The maximally-overlapping speaker label, or ``None`` when
            the window overlaps no turn.
    """
    durations: dict[str, float] = {}
    for turn in turns:
        overlap_start = max(start, float(turn["start"]))
        overlap_end = min(end, float(turn["end"]))
        if overlap_end > overlap_start:
            speaker = str(turn["speaker"])
            durations[speaker] = durations.get(speaker, 0.0) + (overlap_end - overlap_start)
    if not durations:
        return None
    return max(durations, key=lambda s: durations[s])


def assign_speakers_by_overlap(
    transcription_segments: list[dict[str, Any]],
    diarize_segments: list[dict[str, Any]],
) -> None:
    """Label transcript segments with the maximally-overlapping speaker.

    For each transcription segment, the total temporal overlap against every
    diarization turn is accumulated per speaker, and the speaker with the
    greatest overlap wins. Segments that overlap no turn are left untouched
    (they gain no ``speaker`` key). ``transcription_segments`` is mutated in
    place. This is the segment-level fallback used when word timestamps are
    unavailable.

    Args:
        transcription_segments (list[dict[str, Any]]): Whisper segments with
            float ``start`` / ``end`` keys (seconds). Each gains a ``speaker``
            key when an overlapping turn exists.
        diarize_segments (list[dict[str, Any]]): Speaker turns from
            :func:`diarize_file`, each with ``start`` / ``end`` / ``speaker``.
    """
    for segment in transcription_segments:
        speaker = _speaker_by_overlap(float(segment["start"]), float(segment["end"]), diarize_segments)
        if speaker is not None:
            segment["speaker"] = speaker


def build_speaker_segments(
    segments: list[dict[str, Any]],
    words: list[dict[str, Any]],
    turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign speakers to transcript segments, splitting mixed-speaker segments by word.

    When ``words`` is available, each word is assigned the maximally-overlapping
    speaker and every Whisper segment is inspected: if all of its words share a
    single speaker the segment is emitted unchanged (with a ``speaker`` key and
    its **exact** text preserved); if its words carry two or more speakers the
    segment is split at each speaker change, one output segment per run of
    same-speaker words. Words overlapping no turn do not force a split — they
    join the surrounding run.

    When ``words`` is empty (an endpoint that returns no word timestamps), it
    falls back to segment-level :func:`assign_speakers_by_overlap` on copies of
    the input.

    Args:
        segments (list[dict[str, Any]]): Whisper segments with float ``start`` /
            ``end`` and ``text`` keys.
        words (list[dict[str, Any]]): Whisper words with float ``start`` /
            ``end`` and ``word`` keys; may be empty.
        turns (list[dict[str, Any]]): Canonicalized speaker turns.

    Returns:
        list[dict[str, Any]]: New segment dicts, speaker-labeled and — where a
            segment spanned a speaker change — split at the word boundary.
    """
    if not words:
        labeled = [dict(segment) for segment in segments]
        assign_speakers_by_overlap(labeled, turns)
        return labeled

    result: list[dict[str, Any]] = []
    for segment in segments:
        seg_start = float(segment["start"])
        seg_end = float(segment["end"])
        seg_words = [w for w in words if seg_start <= (float(w["start"]) + float(w["end"])) / 2 < seg_end]
        labeled_words = [(w, _speaker_by_overlap(float(w["start"]), float(w["end"]), turns)) for w in seg_words]
        distinct = {speaker for _, speaker in labeled_words if speaker is not None}

        if len(distinct) <= 1:
            new_segment = dict(segment)
            speaker = next(iter(distinct), None)
            if speaker is None:
                speaker = _speaker_by_overlap(seg_start, seg_end, turns)
            if speaker is not None:
                new_segment["speaker"] = speaker
            result.append(new_segment)
            continue

        run_words: list[dict[str, Any]] = []
        run_speaker: str | None = None
        for word, speaker in labeled_words:
            if run_words and speaker is not None and run_speaker is not None and speaker != run_speaker:
                result.append(_word_run_segment(run_words, run_speaker))
                run_words = []
                run_speaker = None
            run_words.append(word)
            if speaker is not None:
                run_speaker = speaker
        if run_words:
            result.append(_word_run_segment(run_words, run_speaker))
    return result


def _word_run_segment(run_words: list[dict[str, Any]], speaker: str | None) -> dict[str, Any]:
    """Build one output segment from a run of same-speaker words.

    Args:
        run_words (list[dict[str, Any]]): Consecutive Whisper words with
            ``word`` / ``start`` / ``end`` keys.
        speaker (str | None): The run's speaker label, if any.

    Returns:
        dict[str, Any]: A segment with ``start`` / ``end`` / ``text`` and an
            optional ``speaker`` key. Text joins the words with single spaces
            (imperfect for space-less scripts — a documented limitation, and
            only hit for genuinely mixed-speaker segments).
    """
    segment: dict[str, Any] = {
        "start": float(run_words[0]["start"]),
        "end": float(run_words[-1]["end"]),
        "text": " ".join(str(w["word"]).strip() for w in run_words).strip(),
    }
    if speaker is not None:
        segment["speaker"] = speaker
    return segment
