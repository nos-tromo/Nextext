"""Tests for ``nextext.core.words.WordCounter`` edge cases."""

from collections import Counter
from typing import Any

import pytest

from nextext.core import words
from nextext.core.words import WordCounter


def test_create_wordcloud_returns_none_when_word_counts_empty() -> None:
    """Empty ``word_counts`` must yield ``None`` rather than crashing.

    The underlying ``wordcloud`` library raises ``ValueError`` when given an
    empty text payload. ``create_wordcloud`` should detect this case (e.g. a
    very short transcript whose tokens were all filtered as stopwords) and
    return ``None`` so the calling pipeline can skip rendering gracefully.
    """
    wc = WordCounter.__new__(WordCounter)
    wc.language = "de"
    wc.font_path = None  # type: ignore[assignment]
    wc.word_counts = Counter()

    assert wc.create_wordcloud() is None


def _make_counter(text: str) -> WordCounter:
    """Build a WordCounter without running spaCy model loading.

    Args:
        text (str): The text under analysis.

    Returns:
        WordCounter: A minimally-initialised instance for NER tests.
    """
    wc = WordCounter.__new__(WordCounter)
    wc.text = text
    wc.language = "en"
    return wc


def test_named_entity_recognition_aggregates_remote_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote entities are filtered, upper-cased, and counted into the DataFrame.

    Args:
        monkeypatch (pytest.MonkeyPatch): Substitutes the remote extractor
            factory with a stub.
    """
    entities = [
        {"text": "Berlin", "type": "loc", "score": 0.9},
        {"text": "Berlin", "type": "loc", "score": 0.8},
        {"text": "Al", "type": "person", "score": 0.9},  # < 3 chars — dropped
        {"text": "Alice", "type": "", "score": 0.9},  # empty label — dropped
    ]
    monkeypatch.setattr(words, "build_remote_ner_extractor", lambda: lambda _chunk: entities)
    wc = _make_counter("Alice flew to Berlin.")

    df = wc.named_entity_recognition()

    assert list(df.columns) == ["Category", "Entity", "Frequency"]
    assert df.to_dict(orient="records") == [{"Category": "LOC", "Entity": "Berlin", "Frequency": 2}]


def test_named_entity_recognition_calls_extractor_per_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The extractor is built once and invoked once per text chunk.

    Args:
        monkeypatch (pytest.MonkeyPatch): Substitutes chunking and the
            remote extractor factory.
    """
    factory_calls = {"n": 0}
    seen_chunks: list[str] = []

    def _factory() -> Any:
        factory_calls["n"] += 1

        def _extract(chunk: str) -> list[dict[str, Any]]:
            seen_chunks.append(chunk)
            return [{"text": f"Entity{len(seen_chunks)}", "type": "org", "score": 0.9}]

        return _extract

    monkeypatch.setattr(words, "build_remote_ner_extractor", _factory)
    monkeypatch.setattr(words, "_chunk_text", lambda _text: ["chunk one", "chunk two", "chunk three"])
    wc = _make_counter("long text")

    df = wc.named_entity_recognition()

    assert factory_calls["n"] == 1
    assert seen_chunks == ["chunk one", "chunk two", "chunk three"]
    assert len(df) == 3


def test_named_entity_recognition_empty_text_skips_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blank text yields an empty DataFrame without building the extractor.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fails the test if the extractor
            factory is consulted.
    """

    def _must_not_run() -> Any:
        raise AssertionError("extractor must not be built for empty text")

    monkeypatch.setattr(words, "build_remote_ner_extractor", _must_not_run)
    wc = _make_counter("   ")

    df = wc.named_entity_recognition()

    assert df.empty
    assert list(df.columns) == ["Category", "Entity", "Frequency"]


def test_named_entity_recognition_no_entities_returns_empty_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fail-soft extractor returning [] everywhere yields an empty DataFrame.

    Args:
        monkeypatch (pytest.MonkeyPatch): Substitutes the remote extractor
            factory with one returning no entities.
    """
    monkeypatch.setattr(words, "build_remote_ner_extractor", lambda: lambda _chunk: [])
    wc = _make_counter("Some text without entities.")

    df = wc.named_entity_recognition()

    assert df.empty
