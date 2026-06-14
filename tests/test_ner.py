"""Tests for the named-entity-recognition agent (HTTP /gliner client)."""

from typing import Any

import httpx
import pytest

from nextext.core import ner
from nextext.core.ner import extract_entities


def test_extract_entities_returns_empty_when_base_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no NER endpoint configured (dedicated or central), NER issues no request.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
    """
    monkeypatch.delenv("NER_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    def fail_post(url: str, **kwargs: Any) -> httpx.Response:
        raise AssertionError("httpx.post must not be called when no NER endpoint is configured")

    monkeypatch.setattr(ner.httpx, "post", fail_post)

    df = extract_entities("Barack Obama visited Berlin.")

    assert list(df.columns) == ["Category", "Entity", "Frequency"]
    assert df.empty


def test_extract_entities_posts_correctly_and_parses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The request targets /gliner with text + labels, bearer auth, and timeout.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
    """
    monkeypatch.setenv("NER_API_BASE", "http://router:4000/")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.delenv("NER_TIMEOUT", raising=False)
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured["url"] = url
        captured.update(kwargs)
        return httpx.Response(
            200,
            json={
                "entities": [
                    {"start": 0, "end": 12, "text": "Barack Obama", "label": "person", "score": 0.99},
                    {"start": 21, "end": 27, "text": "Berlin", "label": "loc", "score": 0.97},
                ]
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(ner.httpx, "post", fake_post)

    df = extract_entities("Barack Obama visited Berlin.")

    assert captured["url"] == "http://router:4000/gliner"
    assert captured["json"]["text"] == "Barack Obama visited Berlin."
    assert captured["json"]["labels"] == ner._NER_LABELS
    assert captured["headers"]["Authorization"] == "Bearer sk-secret"
    assert captured["timeout"] == 120.0
    rows = {(c, e): f for c, e, f in df.itertuples(index=False)}
    assert rows[("PERSON", "Barack Obama")] == 1
    assert rows[("LOC", "Berlin")] == 1


def test_extract_entities_omits_authorization_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No Authorization header is sent when OPENAI_API_KEY is empty.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
    """
    monkeypatch.setenv("NER_API_BASE", "http://router:4000")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured.update(kwargs)
        return httpx.Response(200, json={"entities": []}, request=httpx.Request("POST", url))

    monkeypatch.setattr(ner.httpx, "post", fake_post)

    extract_entities("Some text with no entities of interest.")

    assert "Authorization" not in captured["headers"]


def test_extract_entities_filters_low_score_and_short_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entities below the score threshold or shorter than 3 chars are dropped.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
    """
    monkeypatch.setenv("NER_API_BASE", "http://router:4000")

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "entities": [
                    {"text": "Berlin", "label": "loc", "score": 0.9},
                    {"text": "Bonn", "label": "loc", "score": 0.1},
                    {"text": "UN", "label": "org", "score": 0.95},
                ]
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(ner.httpx, "post", fake_post)

    df = extract_entities("Berlin Bonn UN are words.")

    pairs = {(c, e) for c, e, _ in df.itertuples(index=False)}
    assert pairs == {("LOC", "Berlin")}


def test_extract_entities_chunks_long_text_into_multiple_posts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Text beyond the word budget is split across multiple /gliner requests.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
    """
    monkeypatch.setenv("NER_API_BASE", "http://router:4000")
    posts: list[str] = []

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        posts.append(kwargs["json"]["text"])
        return httpx.Response(200, json={"entities": []}, request=httpx.Request("POST", url))

    monkeypatch.setattr(ner.httpx, "post", fake_post)
    long_text = ("word " * 600).strip() + ". " + ("term " * 600).strip() + "."

    extract_entities(long_text)

    assert len(posts) >= 2


def test_extract_entities_swallows_http_status_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-2xx response is logged and yields an empty DataFrame.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
    """
    monkeypatch.setenv("NER_API_BASE", "http://router:4000")

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(500, text="boom", request=httpx.Request("POST", url))

    monkeypatch.setattr(ner.httpx, "post", fake_post)

    df = extract_entities("Barack Obama visited Berlin.")

    assert df.empty


def test_extract_entities_swallows_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transport error (e.g. connection refused) is logged and yields an empty DataFrame.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
    """
    monkeypatch.setenv("NER_API_BASE", "http://router:4000")

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(ner.httpx, "post", fake_post)

    df = extract_entities("Barack Obama visited Berlin.")

    assert df.empty


def test_extract_entities_handles_non_dict_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A JSON payload that is not an object is rejected and yields an empty DataFrame.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
    """
    monkeypatch.setenv("NER_API_BASE", "http://router:4000")

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json=["not", "a", "dict"], request=httpx.Request("POST", url))

    monkeypatch.setattr(ner.httpx, "post", fake_post)

    df = extract_entities("Barack Obama visited Berlin.")

    assert df.empty


def test_extract_entities_handles_non_numeric_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-numeric score is dropped gracefully without crashing.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
    """
    monkeypatch.setenv("NER_API_BASE", "http://router:4000")

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "entities": [
                    {"text": "Berlin", "label": "loc", "score": "not-a-number"},
                    {"text": "Munich", "label": "loc", "score": 0.9},
                ]
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(ner.httpx, "post", fake_post)

    df = extract_entities("Berlin and Munich.")

    pairs = {(c, e) for c, e, _ in df.itertuples(index=False)}
    assert pairs == {("LOC", "Munich")}


def test_extract_entities_aggregates_entities_across_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same entity returned by multiple chunks is merged into one tallied row.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
    """
    monkeypatch.setenv("NER_API_BASE", "http://router:4000")

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200,
            json={"entities": [{"text": "Berlin", "label": "loc", "score": 0.9}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(ner.httpx, "post", fake_post)
    long_text = ("word " * 600).strip() + ". " + ("term " * 600).strip() + "."

    df = extract_entities(long_text)

    assert len(df) == 1
    rows = {(c, e): f for c, e, f in df.itertuples(index=False)}
    assert rows[("LOC", "Berlin")] == 2


def test_extract_entities_empty_text_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whitespace-only text yields an empty table without issuing a request.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
    """
    monkeypatch.setenv("NER_API_BASE", "http://router:4000")

    def fail_post(url: str, **kwargs: Any) -> httpx.Response:
        raise AssertionError("httpx.post must not be called for blank text")

    monkeypatch.setattr(ner.httpx, "post", fail_post)

    df = extract_entities("   ")

    assert list(df.columns) == ["Category", "Entity", "Frequency"]
    assert df.empty
