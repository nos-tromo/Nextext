"""Shared pipeline entry points for Nextext processing stages."""

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from matplotlib.figure import Figure

from nextext.core.diarization import (
    build_speaker_segments,
    canonicalize_speaker_labels,
    diarize_file,
    gate_turns_by_vad,
)
from nextext.core.hate_speech import HateSpeechDetector
from nextext.core.ner import extract_entities
from nextext.core.openai_cfg import InferencePipeline
from nextext.core.transcription import ExternalWhisperTranscriber
from nextext.core.translation import Translator
from nextext.core.vad import speech_segments
from nextext.core.words import WordCounter
from nextext.utils.env_cfg import load_diarize_vad_gate_env, load_summary_env, load_whisper_env


def transcription_pipeline(
    file_path: Path,
    src_lang: str,
    diarize: bool,
) -> tuple[pd.DataFrame, str]:
    """Transcribe the audio file via the external Whisper API, optionally diarized.

    The audio always goes to an OpenAI-compatible ``/v1/audio/transcriptions``
    endpoint resolved by :func:`nextext.utils.env_cfg.load_whisper_env`;
    Nextext ships no local Whisper. Whisper always transcribes in the source
    language — translation to a target language is handled separately by
    :func:`translation_pipeline`.

    When ``diarize`` is true and the transcript is non-empty, the audio is sent
    to the ``/diarize`` service with **no** speaker bounds (pyannote estimates
    the count). The returned turns are gated by the ``/vad`` speech timeline
    (when ``NEXTEXT_DIARIZE_VAD_GATE`` is on, the default): each turn is cropped
    to VAD speech, so music/noise the diarizer over-detects as speech is dropped
    — a VAD outage leaves the turns ungated. The gated turns are then relabeled
    to contiguous ``Speaker N`` by first appearance and aligned onto the
    transcript at the word level, so a Whisper segment spanning a speaker change
    is split at the exact word. It degrades to segment-level alignment when the
    endpoint returns no words, and to an unlabelled transcript when
    ``DIARIZE_API_BASE`` is unset or the service is unreachable. Diarization is
    skipped for ``diarize=False`` and for empty transcripts.

    Args:
        file_path (Path): Path to the audio file.
        src_lang (str): Source language code.
        diarize (bool): Whether to run speaker diarization.

    Returns:
        tuple[pd.DataFrame, str]: The transcript DataFrame and the
            resolved source language code.
    """
    config = load_whisper_env()
    transcriber = ExternalWhisperTranscriber(
        file_path=file_path,
        src_lang=src_lang,
        model_id=config.model,
    )
    transcriber.transcription()

    result = transcriber.transcription_result or {}
    segments: list[dict[str, Any]] = result.get("segments", [])
    words: list[dict[str, Any]] = result.get("words", [])
    if diarize and segments and transcriber.transcription_result is not None:
        turns = diarize_file(file_path)
        gate = load_diarize_vad_gate_env()
        if turns and gate.enabled:
            vad_intervals = speech_segments(file_path, threshold=gate.threshold, pad_ms=gate.pad_ms)
            if vad_intervals is not None:
                turns = gate_turns_by_vad(turns, vad_intervals)
        turns = canonicalize_speaker_labels(turns)
        if turns:
            transcriber.transcription_result["segments"] = build_speaker_segments(segments, words, turns)

    df = transcriber.transcript_output()
    updated_src_lang = transcriber.src_lang or src_lang
    return df, updated_src_lang


def normalize_language_code(lang_code: str | None) -> str | None:
    """Collapse a locale/script code to its base language code.

    Args:
        lang_code (str | None): The language code to normalize, e.g. "en-US" or "de-CH".

    Returns:
        str | None: The base language code, e.g. "en" or "de", or None if the input was None.
    """
    if lang_code is None:
        return None
    return lang_code.split("-", 1)[0]


def should_translate(task: str, src_lang: str | None, trg_lang: str) -> bool:
    """Decide whether the LLM translation stage should run.

    Whisper only transcribes, so every translation is performed downstream by
    the LLM (``TEXT_MODEL``). Translation runs when a ``translate`` task was
    requested and the resolved source language differs from the target: a
    same-language request is a no-op, and an English target is translated like
    any other (there is no longer a Whisper audio-translate hop to English).

    Args:
        task (str): The requested task, ``"transcribe"`` or ``"translate"``.
        src_lang (str | None): The resolved source language code.
        trg_lang (str): The target language code.

    Returns:
        bool: ``True`` when the translation stage should run.
    """
    if task != "translate":
        return False
    return normalize_language_code(src_lang) != normalize_language_code(trg_lang)


def translation_pipeline(
    df: pd.DataFrame,
    trg_lang: str,
    src_lang: str | None = None,
    inference_pipeline: InferencePipeline | None = None,
) -> pd.DataFrame:
    """Translate the transcribed text using a machine translation model.

    Translation is performed only if the target language is different from
    the detected source language. The original transcribed text is preserved
    in the ``text`` column; the translated text is written to a separate
    ``translation`` column so both can be cross-referenced in the output
    table (CSV/XLSX exports and the UI transcript table).

    Args:
        df (pd.DataFrame): DataFrame containing the transcribed text.
        trg_lang (str): Target language code for translation.
        src_lang (str | None): Source language code, if already known.
        inference_pipeline (InferencePipeline | None): Shared inference client.

    Returns:
        pd.DataFrame: DataFrame with an added ``translation`` column holding
            the translated text, or unchanged when the source and target
            languages already match.
    """
    translator = Translator(inference_pipeline=inference_pipeline)
    resolved_src_lang = src_lang
    if resolved_src_lang is None:
        detected_lang = translator.detect_language(" ".join(df["text"].astype(str).tolist()))
        resolved_src_lang = detected_lang.get("code")
    if normalize_language_code(resolved_src_lang) == normalize_language_code(trg_lang):
        return df
    df["translation"] = df["text"].apply(lambda text: translator.translate(trg_lang, text, src_lang=resolved_src_lang))
    return df


def effective_text_column(df: pd.DataFrame) -> str:
    """Return the column holding the text downstream agents should analyze.

    The transcript DataFrame always keeps the original transcribed text in
    ``text``. When :func:`translation_pipeline` has run, the translated text
    lives in a separate ``translation`` column, and downstream agents
    (word-level analysis, summarization, hate-speech detection) should
    analyze that translated text rather than the original — matching the
    pre-existing behavior from before translation had its own column.

    Args:
        df (pd.DataFrame): Transcript DataFrame, optionally translated.

    Returns:
        str: ``"translation"`` when present, otherwise ``"text"``.
    """
    return "translation" if "translation" in df.columns else "text"


_TXT_BANNER_RULE = "=" * 40


def _render_transcript_block(df: pd.DataFrame, text_column: str) -> str:
    """Render one transcript text column as readable timestamped blocks.

    Each segment's timestamp/speaker header is fenced above and below by a rule
    line so it stands out from the text body — otherwise a blank line *inside* a
    segment (a paragraph break in the transcribed text) reads just like the
    blank line separating segments. The header is ``[{start} - {end}]``, with
    ``  {speaker}`` appended only when the row carries a speaker label (i.e. the
    job was diarized); an undiarized transcript (no ``speaker`` column) carries
    no speaker suffix. One segment renders as::

        ========================================
        [{start} - {end}]  {speaker}
        ========================================
        {text, which may span multiple paragraphs}

    Segments are separated by a blank line; the whole block ends with a single
    trailing newline.

    Args:
        df (pd.DataFrame): Transcript DataFrame with ``start``/``end`` columns,
            ``text_column``, and an optional ``speaker`` column.
        text_column (str): Column whose text is rendered — ``"text"`` for the
            transcript, ``"translation"`` for the translation.

    Returns:
        str: The rendered blocks (single trailing newline included); ``""`` when
            the frame has no rows.
    """
    has_speaker = "speaker" in df.columns
    blocks: list[str] = []
    for row in df.to_dict("records"):
        header = f"[{row.get('start', '')} - {row.get('end', '')}]"
        if has_speaker:
            speaker = row.get("speaker")
            if not pd.isna(speaker) and str(speaker).strip():
                header = f"{header}  {speaker}"
        raw_text = row.get(text_column, "")
        text = "" if pd.isna(raw_text) else str(raw_text)
        blocks.append(f"{_TXT_BANNER_RULE}\n{header}\n{_TXT_BANNER_RULE}\n{text}")
    return "\n\n".join(blocks) + "\n" if blocks else ""


def transcript_txt_exports(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Split a transcript DataFrame into readable per-segment TXT exports.

    The transcript keeps the original text in ``text`` and, after
    :func:`translation_pipeline`, the translated text in a separate
    ``translation`` column. A single wide table pairing both is hard for a
    customer to read, so this returns one human-readable block export per text
    column (see :func:`_render_transcript_block` for the layout):

    - Transcribe-only frame (no ``translation`` column): a single
      ``("transcript", <blocks>)`` pair.
    - Translated frame: two pairs — ``("transcript", <blocks>)`` and
      ``("translation", <blocks>)`` — so the source and the translation each
      read as their own clean, timestamped document.

    Args:
        df (pd.DataFrame): Transcript DataFrame with ``start``/``end``/``text``
            columns, an optional ``speaker`` column, and an optional
            ``translation`` column.

    Returns:
        list[tuple[str, str]]: ``(label, rendered_blocks)`` pairs, ``"transcript"``
            first and ``"translation"`` second when present.
    """
    exports: list[tuple[str, str]] = [("transcript", _render_transcript_block(df, "text"))]
    if "translation" in df.columns:
        exports.append(("translation", _render_transcript_block(df, "translation")))
    return exports


SUMMARY_MAX_OUTPUT_TOKENS: int = 1024
"""Hard cap on summary output tokens, so generation never crowds out the prompt."""

_CHARS_PER_TOKEN: float = 3.0
"""Conservative characters-per-token estimate used to turn the token budget into a
character budget. Smaller is safer; token-dense scripts (CJK) may need a lower
``SUMMARY_MAX_INPUT_TOKENS`` to compensate."""

_MAX_REDUCE_DEPTH: int = 5
"""Recursion ceiling for the reduce step; a backstop that effectively never fires
because partial summaries shrink the text logarithmically."""

_MAX_OVERFLOW_RETRIES: int = 4
"""How many times to halve the budget and retry when a request still overflows the
model's context window (the character heuristic under-counted, e.g. for CJK)."""

_OVERFLOW_BUDGET_BACKOFF: float = 0.5
"""Multiplier applied to the character budget on each context-overflow retry."""

_CONTEXT_LENGTH_ERROR_MARKERS: tuple[str, ...] = (
    "context length",
    "context_length",
    "context window",
    "maximum context",
    "too many tokens",
    "reduce the length",
    "exceeds the maximum",
)
"""Lower-cased substrings that identify a context-window-overflow error across
providers (vLLM, OpenAI-compatible, Ollama)."""


def _split_to_budget(text: str, char_budget: int) -> list[str]:
    """Split text into chunks no larger than ``char_budget`` characters.

    Words are kept whole and packed greedily; a single token longer than the
    budget is hard-sliced so the function always makes progress and never
    emits a chunk larger than the budget.

    Args:
        text (str): The text to split.
        char_budget (int): Maximum characters per chunk (``>= 1``).

    Returns:
        list[str]: Chunks each no longer than ``char_budget``. Whitespace-only
            input yields a single chunk holding the original text.
    """
    words = text.split()
    if not words:
        return [text]

    chunks: list[str] = []
    current = ""
    for word in words:
        # Hard-slice a token that cannot fit in a chunk on its own.
        remainder = word
        while len(remainder) > char_budget:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(remainder[:char_budget])
            remainder = remainder[char_budget:]
        candidate = f"{current} {remainder}" if current else remainder
        if len(candidate) > char_budget:
            chunks.append(current)
            current = remainder
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _summarize_chunk(text: str, inference_pipeline: InferencePipeline) -> str:
    """Summarize a single within-budget chunk of text via the chat model.

    Args:
        text (str): A chunk of text already known to fit the context budget.
        inference_pipeline (InferencePipeline): Shared inference client.

    Returns:
        str: The model's summary of the chunk, capped at
            :data:`SUMMARY_MAX_OUTPUT_TOKENS` output tokens.
    """
    prompt = inference_pipeline.load_prompt("summary").format(text=text)
    return inference_pipeline.call_model(prompt=prompt, num_predict=SUMMARY_MAX_OUTPUT_TOKENS)


def _summarize_within_budget(
    text: str,
    inference_pipeline: InferencePipeline,
    char_budget: int,
    depth: int,
) -> str:
    """Summarize text hierarchically so no single request exceeds the budget.

    A single in-budget chunk is summarized directly; multiple chunks are
    summarized individually (map) and their summaries joined and recursively
    summarized (reduce). A depth guard bounds the recursion for pathological
    inputs whose summaries never shrink.

    Args:
        text (str): The text to summarize.
        inference_pipeline (InferencePipeline): Shared inference client.
        char_budget (int): Maximum characters of text per request.
        depth (int): Current reduce depth, used to bound the recursion.

    Returns:
        str: The summary of the text.
    """
    chunks = _split_to_budget(text, char_budget)
    if len(chunks) == 1:
        return _summarize_chunk(chunks[0], inference_pipeline)
    if depth >= _MAX_REDUCE_DEPTH:
        logger.warning(
            "Summarization exceeded the maximum reduce depth ({}); summarizing the leading chunk only.",
            _MAX_REDUCE_DEPTH,
        )
        return _summarize_chunk(chunks[0], inference_pipeline)
    partial_summaries = [_summarize_chunk(chunk, inference_pipeline) for chunk in chunks]
    combined = "\n\n".join(partial_summaries)
    return _summarize_within_budget(combined, inference_pipeline, char_budget, depth + 1)


def _is_context_length_error(exc: Exception) -> bool:
    """Report whether an exception looks like a context-window overflow.

    Detection is by message text so it works across providers (vLLM,
    OpenAI-compatible, Ollama) without depending on a specific SDK exception
    type.

    Args:
        exc (Exception): The exception raised by an inference call.

    Returns:
        bool: ``True`` when the message matches a known overflow marker.
    """
    message = str(exc).lower()
    return any(marker in message for marker in _CONTEXT_LENGTH_ERROR_MARKERS)


def summarization_pipeline(
    text: str,
    inference_pipeline: InferencePipeline,
) -> str:
    """Summarize transcript text with a context-window-safe map-reduce strategy.

    The transcript is split into chunks that fit the budget configured by
    ``SUMMARY_MAX_INPUT_TOKENS`` (see
    :func:`nextext.utils.env_cfg.load_summary_env`). Short transcripts are
    summarized in a single request; longer ones are summarized chunk-by-chunk
    and their partial summaries are recursively summarized, so no single request
    can overflow the chat model's context window. Every request caps generation
    at :data:`SUMMARY_MAX_OUTPUT_TOKENS` output tokens.

    If a request still overflows the context window (the character heuristic
    under-counted, e.g. for token-dense scripts), the budget is halved and the
    summary retried up to :data:`_MAX_OVERFLOW_RETRIES` times; if even the
    smallest budget overflows, it degrades to an empty summary rather than
    crashing the job (fail-soft, mirroring the NER/diarization clients).

    Args:
        text (str): The text to summarize.
        inference_pipeline (InferencePipeline): An inference pipeline for language model interactions.

    Returns:
        str: The summarized text, or an empty string if every retry still
            overflowed the context window (a fail-soft degrade, logged as a
            warning).

    Raises:
        ValueError: If the input text is empty.
    """
    if not text:
        raise ValueError("Text cannot be empty.")
    char_budget = int(load_summary_env().max_input_tokens * _CHARS_PER_TOKEN)
    for attempt in range(_MAX_OVERFLOW_RETRIES + 1):
        try:
            return _summarize_within_budget(text, inference_pipeline, char_budget, depth=0)
        except Exception as exc:  # context-length overflows are retried; other errors re-raise
            if not _is_context_length_error(exc):
                raise
            if attempt == _MAX_OVERFLOW_RETRIES:
                logger.warning(
                    "Summarization still overflowed the context window after {} retries; "
                    "returning an empty summary. Lower SUMMARY_MAX_INPUT_TOKENS for this input.",
                    _MAX_OVERFLOW_RETRIES,
                )
                return ""
            shrunk_budget = max(1, int(char_budget * _OVERFLOW_BUDGET_BACKOFF))
            logger.warning(
                "Summarization overflowed the context window; retrying with a smaller budget ({} -> {} chars).",
                char_budget,
                shrunk_budget,
            )
            char_budget = shrunk_budget
    return ""


def wordlevel_pipeline(
    data: pd.DataFrame,
    language: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Figure | None]:
    """Calculate word statistics, named entities, and create a word cloud.

    Analyzes the translated text (``translation`` column) when translation
    has run, otherwise the original transcribed text (``text`` column) — see
    :func:`effective_text_column`.

    Args:
        data (pd.DataFrame): DataFrame containing the text data to analyze.
        language (str): Language code of the text data.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, Figure | None]: A tuple containing
            the word-counts DataFrame, the named-entities DataFrame, and the
            word-cloud figure (or ``None`` when there are no word counts to
            plot).
    """
    text_column = effective_text_column(data)
    word_analysis = WordCounter(
        text=" ".join(data[text_column].astype(str).tolist()),
        language=language,
    )

    word_analysis.text_to_doc()
    word_analysis.lemmatize_doc()
    word_counts = word_analysis.count_words()
    named_entities = extract_entities(word_analysis.text)
    wordcloud = word_analysis.create_wordcloud()

    return word_counts, named_entities, wordcloud


def hate_speech_pipeline(
    df: pd.DataFrame,
    inference_pipeline: InferencePipeline,
    max_chars: int = 2048,
) -> list[dict[str, Any]]:
    """Detect hate speech in each transcript segment using an LLM.

    Only segments flagged as hate speech are included in the returned list.
    Each entry in the list is a :class:`HateSpeechDetection` dict extended with
    the analyzed text for display purposes. Detection runs against the
    translated text (``translation`` column) when translation has run,
    otherwise the original transcribed text (``text`` column) — see
    :func:`effective_text_column`.

    Args:
        df (pd.DataFrame): Transcript DataFrame with a ``text`` column.
        inference_pipeline (InferencePipeline): Shared inference client.
        max_chars (int): Maximum characters per segment sent for detection. Defaults to 2048.

    Returns:
        list[dict]: Flagged segments, each containing hate_speech, category,
            confidence, reason, text, and start (segment timestamp when available).
    """
    detector = HateSpeechDetector(inference_pipeline, max_chars)
    has_start = "start" in df.columns
    text_column = effective_text_column(df)
    results: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        detection = detector.detect(str(row[text_column]))
        if detection["hate_speech"]:
            entry = dict(detection)
            entry["text"] = str(row[text_column])
            entry["start"] = str(row["start"]) if has_start else ""
            results.append(entry)
    return results
