"""Tests for the windowed, stance-aware hate-speech detection agent."""

import json
import re
from pathlib import Path
from typing import Any, cast

import pytest

from nextext.core import hate_speech as hate_speech_module
from nextext.core.hate_speech import (
    CONFIDENCE_LEVELS,
    HATE_SPEECH_CATEGORIES,
    HATE_SPEECH_STANCES,
    TranscriptLine,
    TranscriptWindow,
    classify_window,
    next_window,
    parse_window_reply,
    render_line,
    render_window_prompt,
    window_response_format,
)
from nextext.core.openai_cfg import InferencePipeline


def _item(index: Any, stance: str, **overrides: Any) -> dict[str, Any]:
    """Build one model-shaped finding item.

    Args:
        index (Any): The ``index`` value exactly as the model would emit it.
        stance (str): The ``stance`` value.
        **overrides (Any): Field overrides (``category``, ``confidence``, ...).

    Returns:
        dict[str, Any]: A complete finding item.
    """
    item: dict[str, Any] = {
        "index": index,
        "target": "[Gruppe]",
        "reason": "kurz",
        "stance": stance,
        "category": "ethnicity",
        "confidence": "high",
    }
    item.update(overrides)
    return item


def _reply(*items: dict[str, Any]) -> str:
    """Serialize finding items the way a schema-constrained model answers.

    Args:
        *items (dict[str, Any]): Finding items.

    Returns:
        str: The JSON reply text.
    """
    return json.dumps({"findings": list(items)})


# ---------------------------------------------------------------------------
# parse_window_reply
# ---------------------------------------------------------------------------


def test_parse_keeps_only_endorsed_items() -> None:
    """Quoting, condemning, analysing and unclear rows are suppressed; endorsed hate is kept."""
    raw = _reply(
        _item(3, "quotes_or_reports"),
        _item(4, "condemns_or_counters"),
        _item(5, "endorses", category="religion", confidence="medium", reason="Hetze"),
        _item(6, "analyzes_or_discusses"),
        _item(7, "unclear"),
    )

    findings = parse_window_reply(raw, allowed=[3, 4, 5, 6, 7])

    assert findings == [{"index": 5, "category": "religion", "confidence": "medium", "reason": "Hetze"}]


def test_parse_condemnation_alone_yields_no_findings() -> None:
    """A condemnation such as "Das ist antisemitisch." never becomes a finding."""
    raw = _reply(_item(7, "condemns_or_counters", category="religion"))

    assert parse_window_reply(raw, allowed=[7]) == []


def test_parse_ignores_a_boolean_hate_speech_field() -> None:
    """The verdict comes from stance only; a stray ``hate_speech`` flag cannot create a finding."""
    raw = _reply(_item(2, "condemns_or_counters", hate_speech="true"), _item(3, "unclear", hate_speech=True))

    assert parse_window_reply(raw, allowed=[2, 3]) == []


def test_parse_unknown_stance_is_not_a_finding() -> None:
    """A stance outside the enum is treated as unclear, not as endorsement."""
    raw = _reply(_item(1, "supports"), _item(2, "yes"))

    assert parse_window_reply(raw, allowed=[1, 2]) == []


def test_parse_drops_indices_outside_the_core() -> None:
    """Endorsed items pointing at context rows (or nowhere) are dropped."""
    raw = _reply(_item(3, "endorses"), _item(99, "endorses"), _item(6, "endorses"))

    findings = parse_window_reply(raw, allowed=[5, 6])

    assert findings is not None
    assert [f["index"] for f in findings] == [6]


def test_parse_first_item_wins_on_duplicate_index() -> None:
    """A later endorsing duplicate cannot override the model's first verdict for a row."""
    raw = _reply(_item(7, "condemns_or_counters"), _item(7, "endorses"))

    assert parse_window_reply(raw, allowed=[7]) == []


def test_parse_sorts_findings_by_index() -> None:
    """Findings come back in transcript order whatever order the model used."""
    raw = _reply(_item(9, "endorses"), _item(4, "endorses"))

    assert [f["index"] for f in parse_window_reply(raw, allowed=[4, 9]) or []] == [4, 9]


@pytest.mark.parametrize(
    ("raw_index", "expected"),
    [(12, [12]), ("12", [12]), ("[12]", [12]), (12.0, [12]), (True, []), (12.5, []), (None, [])],
)
def test_parse_index_coercion(raw_index: Any, expected: list[int]) -> None:
    """Integral numbers and digit strings are accepted; booleans and fractions are not.

    Args:
        raw_index (Any): The index as the model emitted it.
        expected (list[int]): The indices that must survive parsing.
    """
    raw = _reply(_item(raw_index, "endorses"))

    assert [f["index"] for f in parse_window_reply(raw, allowed=[1, 12]) or []] == expected


@pytest.mark.parametrize(
    ("raw_category", "expected"),
    [
        ("extremism", "extremism"),
        ("Sexual_Orientation", "sexual_orientation"),
        ("sexual orientation", "sexual_orientation"),
        ("Nationality", "nationality"),
        ("racism", "other"),
        ("none", "other"),
        ("", "other"),
    ],
)
def test_parse_categories_follow_the_gmf_allow_list(raw_category: str, expected: str) -> None:
    """GMF enum categories survive normalisation; anything else becomes ``other``.

    Args:
        raw_category (str): The category label as returned by the model.
        expected (str): The normalised category.
    """
    raw = _reply(_item(1, "endorses", category=raw_category))

    findings = parse_window_reply(raw, allowed=[1]) or []

    assert findings[0]["category"] == expected


@pytest.mark.parametrize(("raw_confidence", "expected"), [("High", "high"), ("MEDIUM", "medium"), ("very", "low")])
def test_parse_confidence_is_normalised(raw_confidence: str, expected: str) -> None:
    """Confidence is lower-cased; unknown values fall back to ``low``.

    Args:
        raw_confidence (str): The confidence as returned by the model.
        expected (str): The normalised confidence.
    """
    raw = _reply(_item(1, "endorses", confidence=raw_confidence))

    findings = parse_window_reply(raw, allowed=[1]) or []

    assert findings[0]["confidence"] == expected


def test_parse_normalises_stance_spelling() -> None:
    """Case and separator variations of an enum stance are recognised."""
    raw = _reply(_item(1, " Endorses "), _item(2, "quotes-or-reports"))

    assert [f["index"] for f in parse_window_reply(raw, allowed=[1, 2]) or []] == [1]


def test_parse_accepts_a_bare_list_and_a_single_object() -> None:
    """Unconstrained replies may drop the ``findings`` wrapper."""
    bare_list = json.dumps([_item(1, "endorses")])
    single = json.dumps(_item(2, "endorses"))

    assert [f["index"] for f in parse_window_reply(bare_list, allowed=[1]) or []] == [1]
    assert [f["index"] for f in parse_window_reply(single, allowed=[2]) or []] == [2]


def test_parse_extracts_json_from_prose_and_code_fences() -> None:
    """JSON wrapped in prose or a Markdown fence is still found."""
    raw = "Here you go:\n```json\n" + _reply(_item(4, "endorses")) + "\n```\nHope that helps."

    assert [f["index"] for f in parse_window_reply(raw, allowed=[4]) or []] == [4]


def test_parse_ignores_json_inside_reasoning_blocks() -> None:
    """A draft answer inside ``<think>`` must not be parsed as the verdict."""
    raw = "<think>maybe " + _reply(_item(4, "endorses")) + "</think>" + _reply(_item(4, "condemns_or_counters"))

    assert parse_window_reply(raw, allowed=[4]) == []


def test_parse_skips_non_object_items() -> None:
    """Strings and numbers inside ``findings`` are ignored, valid items kept."""
    raw = json.dumps({"findings": ["4", 5, _item(6, "endorses")]})

    assert [f["index"] for f in parse_window_reply(raw, allowed=[4, 5, 6]) or []] == [6]


def test_parse_empty_findings_is_a_parsed_empty_result() -> None:
    """``{"findings": []}`` is a valid answer meaning "nothing to report"."""
    assert parse_window_reply('{"findings": []}', allowed=[1]) == []


@pytest.mark.parametrize(
    "raw",
    ["", "I cannot help with that.", "{not valid json}", '"just a string"', "42", '{"findings": "none"}'],
)
def test_parse_unparseable_reply_returns_none(raw: str) -> None:
    """Replies with no usable findings structure are reported as unparseable, not as clean.

    Args:
        raw (str): An unusable model reply.
    """
    assert parse_window_reply(raw, allowed=[1]) is None


def test_parse_caps_reason_length() -> None:
    """An overlong rationale is cut so one reply cannot bloat the findings."""
    raw = _reply(_item(1, "endorses", reason="x" * 2000))

    findings = parse_window_reply(raw, allowed=[1]) or []

    assert len(findings[0]["reason"]) == 500


# ---------------------------------------------------------------------------
# render_line / render_window_prompt
# ---------------------------------------------------------------------------


def test_render_line_formats_speaker_and_translation_aid() -> None:
    """Speaker-labelled rows get a label; translations go on an indented aid line."""
    line = TranscriptLine(
        index=4, text="Das ist antisemitisch.", speaker="Speaker 2", translation="That is antisemitic."
    )

    assert render_line(line, cap=1000) == "[4] Speaker 2: Das ist antisemitisch.\n    → That is antisemitic."
    assert render_line(TranscriptLine(index=5, text="Weiter."), cap=1000) == "[5] Weiter."


def test_render_line_caps_text_and_translation_separately() -> None:
    """An oversized row is clipped (head by default, tail for preceding context)."""
    line = TranscriptLine(index=3, text="abcdefghij", translation="0123456789")

    assert render_line(line, cap=5) == "[3] abcd…\n    → 0123…"
    assert render_line(line, cap=5, keep_tail=True) == "[3] …ghij\n    → …6789"


def _render_template() -> str:
    """Return a compact template that exposes every placeholder.

    Returns:
        str: The template.
    """
    return "L={language}\nB:\n{context_before}\nS({first_index}-{last_index}):\n{segments}\nA:\n{context_after}"


def test_render_window_prompt_fills_every_block() -> None:
    """Context and core blocks carry the rows' own indices, not list positions."""
    lines = [
        TranscriptLine(index=10, text="Guten Tag.", speaker="Speaker 1"),
        TranscriptLine(
            index=11, text="Das ist antisemitisch.", speaker="Speaker 2", translation="That is antisemitic."
        ),
        TranscriptLine(index=13, text="Wir reden weiter."),
    ]
    window = TranscriptWindow(context_start=0, core_start=1, core_end=2, context_end=3)

    prompt = render_window_prompt(
        _render_template(), lines, window, core_chars=1000, context_chars=1000, language="German"
    )

    assert prompt == (
        "L=German\nB:\n[10] Speaker 1: Guten Tag.\nS(11-11):\n"
        "[11] Speaker 2: Das ist antisemitisch.\n    → That is antisemitic.\nA:\n[13] Wir reden weiter."
    )


def test_render_window_prompt_marks_empty_context_with_a_dash() -> None:
    """A window at the transcript edge renders its missing context block as ``—``."""
    lines = [TranscriptLine(index=0, text="Nur eine Zeile.")]
    window = TranscriptWindow(context_start=0, core_start=0, core_end=1, context_end=1)

    prompt = render_window_prompt(_render_template(), lines, window, core_chars=1000, context_chars=0, language="—")

    assert prompt == "L=—\nB:\n—\nS(0-0):\n[0] Nur eine Zeile.\nA:\n—"


def test_render_window_prompt_never_substitutes_inside_transcript_text() -> None:
    """Placeholder-like text spoken in the transcript is inserted verbatim, once."""
    lines = [TranscriptLine(index=0, text="Er sagte {segments} und {first_index}.")]
    window = TranscriptWindow(context_start=0, core_start=0, core_end=1, context_end=1)

    prompt = render_window_prompt(
        "{segments}|{last_index}", lines, window, core_chars=1000, context_chars=0, language="x"
    )

    assert prompt == "[0] Er sagte {segments} und {first_index}.|0"


# ---------------------------------------------------------------------------
# next_window
# ---------------------------------------------------------------------------


def _uniform_lines(count: int) -> list[TranscriptLine]:
    """Build rows whose rendered form ``[i] xxxx`` costs exactly 9 characters each.

    Args:
        count (int): Number of rows (single-digit indices keep the cost uniform).

    Returns:
        list[TranscriptLine]: The rows.
    """
    return [TranscriptLine(index=i, text="xxxx") for i in range(count)]


def test_next_window_packs_the_core_up_to_the_budget() -> None:
    """Rows are added to the core while their rendered cost fits the budget."""
    lines = _uniform_lines(3)

    first = next_window(lines, 0, core_chars=18, context_chars=0)
    second = next_window(lines, first.core_end, core_chars=18, context_chars=0)

    assert first == TranscriptWindow(context_start=0, core_start=0, core_end=2, context_end=2)
    assert second == TranscriptWindow(context_start=2, core_start=2, core_end=3, context_end=3)


def test_next_window_gives_an_oversized_row_its_own_window() -> None:
    """A row larger than the budget is still labelled, alone, so the sweep always advances."""
    lines = [TranscriptLine(index=0, text="a" * 50), TranscriptLine(index=1, text="xxxx")]

    window = next_window(lines, 0, core_chars=18, context_chars=0)

    assert (window.core_start, window.core_end) == (0, 1)


def test_next_window_adds_context_margins_within_budget() -> None:
    """Neighbouring rows are shown on both sides while the context budget lasts."""
    lines = _uniform_lines(5)

    narrow = next_window(lines, 2, core_chars=9, context_chars=9)
    wide = next_window(lines, 2, core_chars=9, context_chars=18)

    assert narrow == TranscriptWindow(context_start=1, core_start=2, core_end=3, context_end=4)
    assert wide == TranscriptWindow(context_start=0, core_start=2, core_end=3, context_end=5)


def test_next_window_always_includes_the_adjacent_context_rows() -> None:
    """The immediate neighbours are included even when they exceed the context budget."""
    lines = [
        TranscriptLine(index=0, text="x" * 40),
        TranscriptLine(index=1, text="xxxx"),
        TranscriptLine(index=2, text="y" * 40),
    ]

    window = next_window(lines, 1, core_chars=9, context_chars=5)

    assert window == TranscriptWindow(context_start=0, core_start=1, core_end=2, context_end=3)


def test_next_window_stops_context_after_the_budget_is_spent() -> None:
    """Rows beyond an over-budget neighbour are not added."""
    lines = [
        TranscriptLine(index=0, text="zzzz"),
        TranscriptLine(index=1, text="x" * 40),
        TranscriptLine(index=2, text="xxxx"),
    ]

    window = next_window(lines, 2, core_chars=9, context_chars=5)

    assert window.context_start == 1


def test_next_window_without_context_budget_has_no_margins() -> None:
    """``context_chars=0`` yields core-only windows (the rollback switch)."""
    lines = _uniform_lines(3)

    window = next_window(lines, 1, core_chars=9, context_chars=0)

    assert window == TranscriptWindow(context_start=1, core_start=1, core_end=2, context_end=2)


def test_next_window_edges_lack_the_missing_side() -> None:
    """The first window has no preceding context and the last none following."""
    lines = _uniform_lines(3)

    first = next_window(lines, 0, core_chars=9, context_chars=100)
    last = next_window(lines, 2, core_chars=9, context_chars=100)

    assert (first.context_start, first.context_end) == (0, 3)
    assert (last.context_start, last.context_end) == (0, 3)


# ---------------------------------------------------------------------------
# window_response_format / classify_window
# ---------------------------------------------------------------------------


def test_window_response_format_restricts_indices_to_the_core() -> None:
    """The schema only admits core row indices and the closed stance/category enums."""
    response_format = window_response_format([5, 6])

    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["findings"]
    item = schema["properties"]["findings"]["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == {"index", "target", "reason", "stance", "category", "confidence"}
    assert item["properties"]["index"]["enum"] == [5, 6]
    assert "endorses" in item["properties"]["stance"]["enum"]
    assert "none" not in item["properties"]["category"]["enum"]


class _RecordingPipeline:
    """Structural stand-in for ``InferencePipeline`` that records ``call_model`` kwargs."""

    def __init__(self, reply: str) -> None:
        """Store the canned reply.

        Args:
            reply (str): The text every call returns.
        """
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def call_model(self, prompt: str, **kwargs: Any) -> str:
        """Record the call and return the canned reply.

        Args:
            prompt (str): The rendered window prompt.
            **kwargs (Any): Remaining ``call_model`` keyword arguments.

        Returns:
            str: The canned reply.
        """
        self.calls.append({"prompt": prompt, **kwargs})
        return self.reply


def test_classify_window_sends_a_schema_and_no_system_prompt() -> None:
    """A structured call constrains indices to the core, drops the system role, and parses."""
    pipeline = _RecordingPipeline(_reply(_item(6, "endorses", category="religion")))

    findings = classify_window(cast(InferencePipeline, pipeline), "PROMPT", [5, 6], structured=True)

    assert findings == [{"index": 6, "category": "religion", "confidence": "high", "reason": "kurz"}]
    call = pipeline.calls[0]
    assert call["prompt"] == "PROMPT"
    assert call["include_system_prompt"] is False
    assert call["temperature"] == 0.0
    assert call["response_format"]["json_schema"]["schema"]["properties"]["findings"]["items"]["properties"]["index"][
        "enum"
    ] == [5, 6]


def test_classify_window_unstructured_call_omits_the_schema() -> None:
    """The unconstrained fallback sends no ``response_format``."""
    pipeline = _RecordingPipeline('{"findings": []}')

    findings = classify_window(cast(InferencePipeline, pipeline), "PROMPT", [1], structured=False)

    assert findings == []
    assert pipeline.calls[0]["response_format"] is None


def test_classify_window_reports_an_unparseable_reply_as_none() -> None:
    """A reply without a findings structure is unparseable, never silently clean."""
    pipeline = _RecordingPipeline("Sorry, I can't do that.")

    assert classify_window(cast(InferencePipeline, pipeline), "PROMPT", [1], structured=True) is None


# ---------------------------------------------------------------------------
# hate_speech_transcript prompt contract (en + de)
# ---------------------------------------------------------------------------

_PROMPT_DIR = Path(hate_speech_module.__file__).resolve().parents[1] / "utils" / "prompts"


@pytest.mark.parametrize("locale", ["en", "de"])
def test_transcript_prompt_renders_every_block(locale: str) -> None:
    """Each locale's template exposes every placeholder the renderer fills, and nothing is left unfilled.

    Args:
        locale (str): Prompt locale directory.
    """
    template = (_PROMPT_DIR / locale / "hate_speech_transcript.txt").read_text(encoding="utf-8")
    lines = [
        TranscriptLine(index=3, text="Vorher.", speaker="Speaker 1"),
        TranscriptLine(index=4, text="Das ist antisemitisch.", speaker="Speaker 2"),
        TranscriptLine(index=5, text="Nachher.", speaker="Speaker 1"),
    ]
    window = TranscriptWindow(context_start=0, core_start=1, core_end=2, context_end=3)

    prompt = render_window_prompt(template, lines, window, core_chars=1000, context_chars=1000, language="German")

    assert "[3] Speaker 1: Vorher." in prompt
    assert "[4] Speaker 2: Das ist antisemitisch." in prompt
    assert "[5] Speaker 1: Nachher." in prompt
    assert "German" in prompt
    assert "4\u20134" in prompt  # en dash between the core bounds
    assert re.search(r"\{(language|context_before|segments|context_after|first_index|last_index)\}", prompt) is None


@pytest.mark.parametrize("locale", ["en", "de"])
def test_transcript_prompt_names_every_parser_enum_value(locale: str) -> None:
    """Unconstrained replies can only use values the prompt names; the parser enums must all appear.

    Args:
        locale (str): Prompt locale directory.
    """
    template = (_PROMPT_DIR / locale / "hate_speech_transcript.txt").read_text(encoding="utf-8")

    for value in (*HATE_SPEECH_STANCES, *HATE_SPEECH_CATEGORIES, *CONFIDENCE_LEVELS):
        assert value in template, value


@pytest.mark.parametrize("locale", ["en", "de"])
def test_transcript_prompt_keeps_the_instruction_prefix_static(locale: str) -> None:
    """Placeholders only appear in the variable tail, so the long instruction prefix is cacheable.

    Args:
        locale (str): Prompt locale directory.
    """
    template = (_PROMPT_DIR / locale / "hate_speech_transcript.txt").read_text(encoding="utf-8")

    first_placeholder = min(template.index(name) for name in ("{language}", "{context_before}", "{segments}"))

    assert first_placeholder > len(template) * 0.7
