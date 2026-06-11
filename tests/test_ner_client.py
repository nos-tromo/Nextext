"""Tests for the remote GLiNER NER HTTP client."""

import httpx
import pytest
import respx

from nextext.core.ner_client import DEFAULT_NER_LABELS, build_remote_ner_extractor
from nextext.utils.env_cfg import NERClientConfig

_BASE = "http://gliner-only:8000"


def _make_cfg(api_key: str | None = None, threshold: float = 0.3) -> NERClientConfig:
    """Build an explicit client configuration for tests.

    Args:
        api_key (str | None): Bearer token, or ``None`` for no auth header.
        threshold (float): Confidence cutoff sent with each request.

    Returns:
        NERClientConfig: The assembled configuration.
    """
    return NERClientConfig(api_base=_BASE, api_key=api_key, threshold=threshold, timeout=5.0)


@respx.mock
def test_extractor_maps_entities() -> None:
    """A successful response is mapped to {text, type, score} entities."""
    route = respx.post(f"{_BASE}/gliner").mock(
        return_value=httpx.Response(
            200,
            json={
                "entities": [
                    {"text": "Berlin", "label": "loc", "score": 0.91},
                    {"text": "Alice", "label": "person", "score": 0.87},
                ]
            },
        )
    )
    extract = build_remote_ner_extractor(cfg=_make_cfg())

    entities = extract("Alice flew to Berlin.")

    assert entities == [
        {"text": "Berlin", "type": "loc", "score": 0.91},
        {"text": "Alice", "type": "person", "score": 0.87},
    ]
    payload = route.calls.last.request.read()
    assert b'"labels"' in payload
    for label in DEFAULT_NER_LABELS:
        assert label.encode() in payload


@respx.mock
def test_extractor_sends_bearer_header_when_key_set() -> None:
    """A configured API key is carried as a Bearer Authorization header."""
    route = respx.post(f"{_BASE}/gliner").mock(return_value=httpx.Response(200, json={"entities": []}))
    extract = build_remote_ner_extractor(cfg=_make_cfg(api_key="sk-ner"))

    extract("some text")

    assert route.calls.last.request.headers["Authorization"] == "Bearer sk-ner"


@respx.mock
def test_extractor_omits_auth_header_without_key() -> None:
    """Without an API key no Authorization header is sent."""
    route = respx.post(f"{_BASE}/gliner").mock(return_value=httpx.Response(200, json={"entities": []}))
    extract = build_remote_ner_extractor(cfg=_make_cfg(api_key=None))

    extract("some text")

    assert "Authorization" not in route.calls.last.request.headers


@respx.mock
def test_extractor_passes_threshold_and_custom_labels() -> None:
    """The configured threshold and explicit labels land in the request body."""
    route = respx.post(f"{_BASE}/gliner").mock(return_value=httpx.Response(200, json={"entities": []}))
    extract = build_remote_ner_extractor(labels=["person"], cfg=_make_cfg(threshold=0.55))

    extract("some text")

    import json

    body = json.loads(route.calls.last.request.read())
    assert body["threshold"] == 0.55
    assert body["labels"] == ["person"]
    assert body["text"] == "some text"


@respx.mock
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(200, json={"unexpected": "shape"}),
        httpx.Response(200, json={"entities": "not-a-list"}),
        httpx.Response(200, text="not json"),
    ],
)
def test_extractor_fails_soft(response: httpx.Response) -> None:
    """Server errors and malformed payloads degrade to an empty entity list.

    Args:
        response (httpx.Response): The faulty response under test.
    """
    respx.post(f"{_BASE}/gliner").mock(return_value=response)
    extract = build_remote_ner_extractor(cfg=_make_cfg())

    assert extract("some text") == []


@respx.mock
def test_extractor_fails_soft_on_connect_error() -> None:
    """Network-level failures degrade to an empty entity list."""
    respx.post(f"{_BASE}/gliner").mock(side_effect=httpx.ConnectError("refused"))
    extract = build_remote_ner_extractor(cfg=_make_cfg())

    assert extract("some text") == []


@respx.mock
def test_extractor_skips_request_for_blank_text() -> None:
    """Blank input returns [] without hitting the network."""
    route = respx.post(f"{_BASE}/gliner").mock(return_value=httpx.Response(200, json={"entities": []}))
    extract = build_remote_ner_extractor(cfg=_make_cfg())

    assert extract("   ") == []
    assert not route.called


@respx.mock
def test_extractor_drops_incomplete_entities() -> None:
    """Entities missing text or label are filtered out of the result."""
    respx.post(f"{_BASE}/gliner").mock(
        return_value=httpx.Response(
            200,
            json={
                "entities": [
                    {"text": "Berlin", "label": "loc", "score": 0.9},
                    {"text": "", "label": "loc", "score": 0.9},
                    {"label": "person", "score": 0.9},
                    {"text": "Bob", "score": 0.9},
                    "not-a-dict",
                ]
            },
        )
    )
    extract = build_remote_ner_extractor(cfg=_make_cfg())

    assert extract("some text") == [{"text": "Berlin", "type": "loc", "score": 0.9}]
