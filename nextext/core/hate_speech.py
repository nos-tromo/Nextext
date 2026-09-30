"""Hate-speech detection agent: windowed, stance-aware transcript classification.

Transcript rows are judged in context windows: a *core* of consecutive rows the
model labels, framed by read-only neighbouring rows before and after so it can
tell who is speaking, what is being answered, and what a pronoun refers to. The
model returns row indices plus the speaker's stance for every candidate row —
never text — and only rows whose speaker *endorses* group-focused enmity become
findings. Quoting, reporting, condemning or analysing hate is not hate.
"""

import json
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

from nextext.core.openai_cfg import InferencePipeline

HATE_SPEECH_STANCES: tuple[str, ...] = (
    "endorses",
    "quotes_or_reports",
    "condemns_or_counters",
    "analyzes_or_discusses",
    "unclear",
)
"""Speaker stances the model may assign to a candidate row; only ``endorses`` is reported."""

HATE_SPEECH_CATEGORIES: tuple[str, ...] = (
    "race",
    "ethnicity",
    "religion",
    "gender",
    "sexual_orientation",
    "disability",
    "nationality",
    "extremism",
    "other",
)
"""Group-focused-enmity (GMF) categories; unknown labels are normalised to ``other``."""

CONFIDENCE_LEVELS: tuple[str, ...] = ("high", "medium", "low")
"""Allowed confidence values; unknown values are normalised to ``low``."""

HS_MAX_OUTPUT_TOKENS: int = 1024
"""Output-token cap for one window request (a findings list, never prose)."""

_REASON_MAX_CHARS: int = 500
_TRANSLATION_PREFIX: str = "\n    → "
_EMPTY_BLOCK: str = "—"
_PLACEHOLDER_RE = re.compile(r"\{(language|context_before|segments|context_after|first_index|last_index)\}")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE: str = "</think>"
_INDEX_TEXT_RE = re.compile(r"^\[?\s*(\d+)\s*\]?$")


@dataclass(frozen=True)
class TranscriptLine:
    """One transcript row as the classifier sees it.

    Attributes:
        index (int): The row's position in the transcript DataFrame — the
            identifier the model reports back.
        text (str): Original-language text, stripped (never the translation).
        speaker (str | None): Diarization label, or ``None`` when unknown.
        translation (str | None): Translation shown as an aid, or ``None``.
    """

    index: int
    text: str
    speaker: str | None = None
    translation: str | None = None


@dataclass(frozen=True)
class TranscriptWindow:
    """Positions (into the line list) of one classification window.

    The core ``[core_start, core_end)`` is labelled; the margins
    ``[context_start, core_start)`` and ``[core_end, context_end)`` are shown
    as read-only context. Consecutive windows have disjoint cores, so every
    row is labelled exactly once.

    Attributes:
        context_start (int): First context position before the core.
        core_start (int): First labelled position.
        core_end (int): One past the last labelled position.
        context_end (int): One past the last context position after the core.
    """

    context_start: int
    core_start: int
    core_end: int
    context_end: int


class WindowFinding(TypedDict):
    """One row whose speaker endorses group-focused enmity.

    Attributes:
        index (int): Row index (``TranscriptLine.index``) of the finding.
        category (str): One of :data:`HATE_SPEECH_CATEGORIES`.
        confidence (str): One of :data:`CONFIDENCE_LEVELS`.
        reason (str): Short rationale in the prompt's language.
    """

    index: int
    category: str
    confidence: str
    reason: str


def _clip(text: str, limit: int, *, keep_tail: bool = False) -> str:
    """Shorten ``text`` to at most ``limit`` characters, marking the cut with ``…``.

    Args:
        text (str): The text to clip.
        limit (int): Maximum length of the result.
        keep_tail (bool): Keep the end of the text instead of its start.

    Returns:
        str: ``text`` unchanged when it fits, otherwise the clipped text.
    """
    if len(text) <= limit:
        return text
    if limit <= 1:
        return "…"
    return "…" + text[-(limit - 1) :] if keep_tail else text[: limit - 1] + "…"


def render_line(line: TranscriptLine, cap: int, *, keep_tail: bool = False) -> str:
    """Render one row as ``[index] Speaker: text`` plus an optional translation aid line.

    Args:
        line (TranscriptLine): The row to render.
        cap (int): Maximum characters kept of the text and, separately, of the
            translation.
        keep_tail (bool): Clip from the start instead of the end — used for the
            context row just before the core, whose ending is closest to it.

    Returns:
        str: The rendered row; a translation goes on an indented ``→`` line.
    """
    prefix = f"[{line.index}] {line.speaker}: " if line.speaker else f"[{line.index}] "
    rendered = prefix + _clip(line.text, cap, keep_tail=keep_tail)
    if line.translation:
        rendered += _TRANSLATION_PREFIX + _clip(line.translation, cap, keep_tail=keep_tail)
    return rendered


def _line_cost(line: TranscriptLine, cap: int, *, keep_tail: bool = False) -> int:
    """Return the rendered size of a row including its trailing newline.

    Args:
        line (TranscriptLine): The row.
        cap (int): The clip limit it will be rendered with.
        keep_tail (bool): Whether it is rendered tail-first.

    Returns:
        int: Characters the row occupies in the prompt.
    """
    return len(render_line(line, cap, keep_tail=keep_tail)) + 1


def next_window(lines: Sequence[TranscriptLine], start: int, core_chars: int, context_chars: int) -> TranscriptWindow:
    """Build the classification window whose core begins at ``start``.

    The core takes rows while their rendered cost fits ``core_chars``, but
    always at least one row, so a sweep always advances (an oversized row is
    clipped in the prompt and labelled alone). Context margins walk outward
    from the core on both sides: the adjacent row is always included (clipped
    to ``context_chars``) and further rows only while they fit the remaining
    budget. ``context_chars <= 0`` disables the margins.

    Args:
        lines (Sequence[TranscriptLine]): All classifiable rows, in order.
        start (int): Position of the first core row; must be ``< len(lines)``.
        core_chars (int): Character budget of the core.
        context_chars (int): Character budget of each context margin.

    Returns:
        TranscriptWindow: The window's positions.
    """
    core_end = start
    used = 0
    while core_end < len(lines):
        cost = _line_cost(lines[core_end], core_chars)
        if core_end > start and used + cost > core_chars:
            break
        used += cost
        core_end += 1

    context_start = start
    context_end = core_end
    if context_chars > 0:
        remaining = context_chars
        position = start - 1
        while position >= 0 and remaining > 0:
            cost = _line_cost(lines[position], context_chars, keep_tail=True)
            if position < start - 1 and cost > remaining:
                break
            remaining -= cost
            context_start = position
            position -= 1

        remaining = context_chars
        position = core_end
        while position < len(lines) and remaining > 0:
            cost = _line_cost(lines[position], context_chars)
            if position > core_end and cost > remaining:
                break
            remaining -= cost
            position += 1
            context_end = position

    return TranscriptWindow(context_start=context_start, core_start=start, core_end=core_end, context_end=context_end)


def render_window_prompt(
    template: str,
    lines: Sequence[TranscriptLine],
    window: TranscriptWindow,
    *,
    core_chars: int,
    context_chars: int,
    language: str,
) -> str:
    """Fill the transcript prompt template for one window in a single pass.

    Every placeholder is substituted by one regular-expression pass over the
    template, so placeholder-like text spoken in the transcript is inserted
    verbatim and never substituted again.

    Args:
        template (str): Prompt template with ``{language}``,
            ``{context_before}``, ``{segments}``, ``{context_after}``,
            ``{first_index}`` and ``{last_index}`` placeholders.
        lines (Sequence[TranscriptLine]): All classifiable rows, in order.
        window (TranscriptWindow): The window to render.
        core_chars (int): Clip limit for core rows.
        context_chars (int): Clip limit for context rows.
        language (str): Human-readable transcript language.

    Returns:
        str: The rendered prompt. An empty context block renders as ``—``.
    """

    def block(start: int, end: int, cap: int, *, keep_tail: bool = False) -> str:
        """Render rows ``[start, end)`` one per line.

        Args:
            start (int): First position.
            end (int): One past the last position.
            cap (int): Clip limit.
            keep_tail (bool): Clip tail-first.

        Returns:
            str: The rendered rows, or ``—`` when the range is empty.
        """
        rendered = [render_line(lines[position], cap, keep_tail=keep_tail) for position in range(start, end)]
        return "\n".join(rendered) if rendered else _EMPTY_BLOCK

    values = {
        "language": language,
        "context_before": block(window.context_start, window.core_start, context_chars, keep_tail=True),
        "segments": block(window.core_start, window.core_end, core_chars),
        "context_after": block(window.core_end, window.context_end, context_chars),
        "first_index": str(lines[window.core_start].index),
        "last_index": str(lines[window.core_end - 1].index),
    }
    return _PLACEHOLDER_RE.sub(lambda match: values[match.group(1)], template)


def window_response_format(indices: Sequence[int]) -> dict[str, Any]:
    """Build the ``response_format`` JSON schema for one window.

    Portable across vLLM, Ollama and OpenAI strict mode: every property is
    required and no additional properties are allowed. ``index`` is restricted
    to the core rows, so a schema-enforcing backend cannot report a context
    row. ``reason`` precedes ``stance`` both in declaration order and
    alphabetically, so the rationale is generated before the decision on every
    backend.

    Args:
        indices (Sequence[int]): Row indices of the window's core.

    Returns:
        dict[str, Any]: The ``response_format`` payload.
    """
    item_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["index", "target", "reason", "stance", "category", "confidence"],
        "properties": {
            "index": {"type": "integer", "enum": list(indices)},
            "target": {"type": "string"},
            "reason": {"type": "string"},
            "stance": {"type": "string", "enum": list(HATE_SPEECH_STANCES)},
            "category": {"type": "string", "enum": list(HATE_SPEECH_CATEGORIES)},
            "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "hate_speech_findings",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["findings"],
                "properties": {"findings": {"type": "array", "items": item_schema}},
            },
        },
    }


def _strip_reasoning(raw: str) -> str:
    """Remove ``<think>`` reasoning blocks (and anything before a dangling close tag).

    Args:
        raw (str): The raw model reply.

    Returns:
        str: The reply without reasoning, stripped.
    """
    text = _THINK_BLOCK_RE.sub("", raw)
    close = text.lower().rfind(_THINK_CLOSE)
    if close != -1:
        text = text[close + len(_THINK_CLOSE) :]
    return text.strip()


def _looks_like_findings(candidate: Any) -> bool:
    """Report whether a decoded JSON value can stand in for a findings payload.

    Args:
        candidate (Any): A decoded JSON value.

    Returns:
        bool: ``True`` for a single finding object or a list of objects.
    """
    if isinstance(candidate, dict):
        return "index" in candidate
    if isinstance(candidate, list):
        return all(isinstance(item, dict) for item in candidate)
    return False


def _load_json_payload(text: str) -> Any:
    """Decode the findings payload from a reply that may wrap it in prose or fences.

    Args:
        text (str): The reply with reasoning removed.

    Returns:
        Any: The first ``{"findings": ...}`` object, else the first value that
            looks like findings, else the whole reply decoded as JSON (or
            ``None`` when nothing decodes).
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    fallback: Any = None
    for position, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            candidate, _ = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "findings" in candidate:
            return candidate
        if fallback is None and _looks_like_findings(candidate):
            fallback = candidate
    return fallback


def _coerce_index(value: Any) -> int | None:
    """Convert a model-emitted row index to ``int``.

    Args:
        value (Any): The ``index`` value (int, integral float, or digit string,
            optionally in brackets).

    Returns:
        int | None: The index, or ``None`` for booleans, fractions and junk.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        match = _INDEX_TEXT_RE.match(value.strip())
        return int(match.group(1)) if match else None
    return None


def _normalize_choice(value: Any, choices: Sequence[str], default: str) -> str:
    """Normalise an enum-like string (case, whitespace, separators) against ``choices``.

    Args:
        value (Any): The raw value.
        choices (Sequence[str]): Allowed values.
        default (str): Value returned for anything not in ``choices``.

    Returns:
        str: The matching choice, or ``default``.
    """
    if not isinstance(value, str):
        return default
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized if normalized in choices else default


def parse_window_reply(raw: str, allowed: Collection[int]) -> list[WindowFinding] | None:
    """Parse a window reply into the rows whose speaker endorses hate.

    No boolean verdict is read: a row is a finding if and only if its stance
    is ``endorses``. Items for rows outside ``allowed`` are dropped; the first
    item per row wins; categories and confidences are normalised to their
    enums.

    Args:
        raw (str): The raw model reply.
        allowed (Collection[int]): Row indices of the window's core.

    Returns:
        list[WindowFinding] | None: Endorsed-hate findings in row order (``[]``
            when the reply lists none), or ``None`` when the reply holds no
            usable findings structure.
    """
    payload = _load_json_payload(_strip_reasoning(raw or ""))
    items: Any
    if isinstance(payload, dict):
        items = payload.get("findings") if "findings" in payload else ([payload] if "index" in payload else None)
    elif isinstance(payload, list):
        items = payload
    else:
        return None
    if not isinstance(items, list):
        return None

    allowed_set = set(allowed)
    seen: set[int] = set()
    findings: list[WindowFinding] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        index = _coerce_index(item.get("index"))
        if index is None or index not in allowed_set or index in seen:
            continue
        seen.add(index)
        if _normalize_choice(item.get("stance"), HATE_SPEECH_STANCES, "unclear") != "endorses":
            continue
        findings.append(
            WindowFinding(
                index=index,
                category=_normalize_choice(item.get("category"), HATE_SPEECH_CATEGORIES, "other"),
                confidence=_normalize_choice(item.get("confidence"), CONFIDENCE_LEVELS, "low"),
                reason=str(item.get("reason") or "").strip()[:_REASON_MAX_CHARS],
            )
        )
    findings.sort(key=lambda finding: finding["index"])
    return findings


def classify_window(
    inference_pipeline: InferencePipeline,
    prompt: str,
    indices: Sequence[int],
    *,
    structured: bool,
) -> list[WindowFinding] | None:
    """Classify one rendered window and return its endorsed-hate findings.

    The request carries no system message — the prompt holds all instructions
    — and runs at temperature 0. With ``structured`` the reply is constrained
    by :func:`window_response_format`; the parser accepts unconstrained replies
    either way, because routers may drop the constraint silently.

    Args:
        inference_pipeline (InferencePipeline): Shared inference client.
        prompt (str): The rendered window prompt.
        indices (Sequence[int]): Row indices of the window's core.
        structured (bool): Whether to send the JSON-schema constraint.

    Returns:
        list[WindowFinding] | None: Findings, or ``None`` when the reply could
            not be parsed.
    """
    raw = inference_pipeline.call_model(
        prompt=prompt,
        temperature=0.0,
        num_predict=HS_MAX_OUTPUT_TOKENS,
        include_system_prompt=False,
        response_format=window_response_format(indices) if structured else None,
    )
    return parse_window_reply(raw, allowed=indices)
