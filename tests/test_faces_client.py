"""Tests for the faces agent (external ``/faces/detect`` and ``/faces/group`` HTTP client).

Both calls are fail-soft: no endpoint, a transport or HTTP error, or a reply
that does not match the contract resolve to ``None`` so the face stage degrades
instead of failing the job. Images are synthetic bytes; no face is ever sent.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest

from nextext.core import faces
from nextext.core.faces import (
    FACES_MODEL_ALIAS,
    MAX_REQUEST_BYTES,
    DetectedFace,
    GroupFace,
    GroupTrack,
    detect_faces,
    group_faces,
)
from nextext.core.inference_trace import trace_inference

_FACE = {
    "box": [0.41, 0.22, 0.12, 0.21],
    "score": 0.97,
    "landmarks": [[0.44, 0.29], [0.50, 0.29], [0.47, 0.33], [0.45, 0.37], [0.49, 0.37]],
}


@pytest.fixture(autouse=True)
def _no_faces_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each test without any endpoint configured.

    Args:
        monkeypatch: Pytest patcher.
    """
    for name in ("FACES_API_BASE", "FACES_TIMEOUT", "OPENAI_API_BASE", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _reply(url: str, payload: object, status: int = 200) -> httpx.Response:
    """Build an httpx response to a POST.

    Args:
        url: The request URL.
        payload: The JSON body.
        status: The HTTP status.

    Returns:
        httpx.Response: The response, with its request attached.
    """
    return httpx.Response(status, json=payload, request=httpx.Request("POST", url))


def _no_request(url: str, **kwargs: Any) -> httpx.Response:
    """Fail the test if a request is made.

    Args:
        url: The request URL.
        **kwargs: Request arguments.

    Raises:
        AssertionError: Always.
    """
    raise AssertionError(f"no request expected, got one to {url}")


def _track(samples: tuple[int, ...], jpeg: bytes = b"\xff\xd8crop") -> GroupTrack:
    """Build a one-face track for ``/faces/group``.

    Args:
        samples: The sample indices the track spans.
        jpeg: The crop's bytes.

    Returns:
        GroupTrack: The track.
    """
    return GroupTrack(samples=samples, faces=(GroupFace(jpeg=jpeg, landmarks=((10.0, 12.0), (20.0, 12.0))),))


def test_detect_posts_json_with_bearer_and_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    """A detect call posts base64 images to the root's /faces/detect and parses the boxes."""
    monkeypatch.setenv("FACES_API_BASE", "http://router:4000/")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent: list[dict[str, Any]] = []

    def post(url: str, **kwargs: Any) -> httpx.Response:
        sent.append({"url": url, **kwargs})
        return _reply(url, {"results": [{"faces": [_FACE]}, {"faces": []}]})

    monkeypatch.setattr(faces.httpx, "post", post)

    results = detect_faces([b"first-image", b"second-image"])

    assert results == [
        [
            DetectedFace(
                box=(0.41, 0.22, 0.12, 0.21),
                score=0.97,
                landmarks=((0.44, 0.29), (0.50, 0.29), (0.47, 0.33), (0.45, 0.37), (0.49, 0.37)),
            )
        ],
        [],
    ]
    assert len(sent) == 1
    assert sent[0]["url"] == "http://router:4000/faces/detect"
    assert sent[0]["json"] == {
        "images": [base64.b64encode(b"first-image").decode("ascii"), base64.b64encode(b"second-image").decode("ascii")],
        "min_score": 0.7,
    }
    assert sent[0]["headers"]["Authorization"] == "Bearer sk-test"
    assert sent[0]["timeout"] == 300.0


def test_detect_unset_base_with_off_token_makes_no_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """FACES_API_BASE=off switches the client off even with a central endpoint."""
    monkeypatch.setenv("OPENAI_API_BASE", "http://router:4000/v1")
    monkeypatch.setenv("FACES_API_BASE", "off")
    monkeypatch.setattr(faces.httpx, "post", _no_request)

    assert detect_faces([b"image"]) is None
    assert group_faces([_track((0,))]) is None


def test_detect_without_any_endpoint_makes_no_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """With neither a dedicated nor a central endpoint, nothing is sent."""
    monkeypatch.setattr(faces.httpx, "post", _no_request)

    assert detect_faces([b"image"]) is None


def test_detect_falls_back_to_central_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without FACES_API_BASE the central endpoint's root (one /v1 stripped) is used."""
    monkeypatch.setenv("OPENAI_API_BASE", "http://r:4000/v1")
    urls: list[str] = []

    def post(url: str, **kwargs: Any) -> httpx.Response:
        urls.append(url)
        return _reply(url, {"results": [{"faces": []}]})

    monkeypatch.setattr(faces.httpx, "post", post)

    assert detect_faces([b"image"]) == [[]]
    assert urls == ["http://r:4000/faces/detect"]


def test_detect_marks_undecodable_entries_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """An image the service could not decode is None in place; the others keep their faces."""
    monkeypatch.setenv("FACES_API_BASE", "http://router:4000")
    monkeypatch.setattr(
        faces.httpx,
        "post",
        lambda url, **kwargs: _reply(url, {"results": [{"error": "undecodable"}, {"faces": [_FACE]}]}),
    )

    results = detect_faces([b"broken", b"image"])

    assert results is not None
    assert results[0] is None
    assert results[1] is not None and len(results[1]) == 1


def test_detect_result_count_mismatch_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reply with a different number of results than images is a failure."""
    monkeypatch.setenv("FACES_API_BASE", "http://router:4000")
    monkeypatch.setattr(faces.httpx, "post", lambda url, **kwargs: _reply(url, {"results": [{"faces": []}]}))

    assert detect_faces([b"one", b"two"]) is None


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"results": "nope"},
        {"results": [{"faces": [{"box": [0.1, 0.2], "score": 0.9, "landmarks": []}]}]},
        {"results": [{"faces": [{"box": [0.1, 0.2, 0.3, 0.4], "score": "high", "landmarks": []}]}]},
        {"results": [{"faces": [{"box": [0.1, 0.2, 0.3, 0.4], "score": 0.9, "landmarks": [[0.1]]}]}]},
    ],
)
def test_detect_malformed_reply_is_none(monkeypatch: pytest.MonkeyPatch, payload: object) -> None:
    """A reply that does not match the contract is a failure, recorded as one.

    Args:
        monkeypatch: Pytest patcher.
        payload: The malformed reply.
    """
    monkeypatch.setenv("FACES_API_BASE", "http://router:4000")
    monkeypatch.setattr(faces.httpx, "post", lambda url, **kwargs: _reply(url, payload))

    with trace_inference() as trace:
        assert detect_faces([b"image"]) is None
    assert [(t.task, t.model, t.requests, t.failed) for t in trace.tallies] == [
        ("face_detect", FACES_MODEL_ALIAS, 1, 1)
    ]


def test_detect_http_error_is_none_and_logged_by_class(monkeypatch: pytest.MonkeyPatch, log_lines: list[str]) -> None:
    """A rejection whose body echoes the request is logged by class and status only."""
    monkeypatch.setenv("FACES_API_BASE", "http://router:4000")
    echoed = "invented upstream detail 0427"
    monkeypatch.setattr(
        faces.httpx,
        "post",
        lambda url, **kwargs: httpx.Response(500, text=f"failed: {echoed}", request=httpx.Request("POST", url)),
    )

    assert detect_faces([b"image"]) is None

    logged = "\n".join(log_lines)
    assert "HTTPStatusError (HTTP 500)" in logged
    assert echoed not in logged
    assert base64.b64encode(b"image").decode("ascii") not in logged


def test_detect_transport_error_is_none(monkeypatch: pytest.MonkeyPatch, log_lines: list[str]) -> None:
    """A transport error is named by class; its message is not logged."""
    monkeypatch.setenv("FACES_API_BASE", "http://router:4000")
    detail = "invented transport detail 0427"

    def post(url: str, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError(detail)

    monkeypatch.setattr(faces.httpx, "post", post)

    assert detect_faces([b"image"]) is None
    logged = "\n".join(log_lines)
    assert "ConnectError" in logged
    assert detail not in logged


def test_detect_reports_one_request_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each detect request is tallied once under face_detect, however many images it carries."""
    monkeypatch.setenv("FACES_API_BASE", "http://router:4000")
    monkeypatch.setattr(
        faces.httpx, "post", lambda url, **kwargs: _reply(url, {"results": [{"faces": []}, {"faces": []}]})
    )

    with trace_inference() as trace:
        detect_faces([b"one", b"two"])
    assert [(t.task, t.model, t.requests, t.failed) for t in trace.tallies] == [
        ("face_detect", FACES_MODEL_ALIAS, 1, 0)
    ]


def test_group_returns_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """A group call posts each track's samples and crops and returns one label per track."""
    monkeypatch.setenv("FACES_API_BASE", "http://router:4000")
    sent: list[dict[str, Any]] = []

    def post(url: str, **kwargs: Any) -> httpx.Response:
        sent.append({"url": url, **kwargs})
        return _reply(url, {"labels": [0, 0, -1]})

    monkeypatch.setattr(faces.httpx, "post", post)

    with trace_inference() as trace:
        labels = group_faces([_track((0, 1)), _track((3,)), _track((4, 5), jpeg=b"\xff\xd8other")])

    assert labels == [0, 0, -1]
    assert sent[0]["url"] == "http://router:4000/faces/group"
    body = json.loads(sent[0]["content"])
    assert body["tracks"][0] == {
        "samples": [0, 1],
        "faces": [
            {"image": base64.b64encode(b"\xff\xd8crop").decode("ascii"), "landmarks": [[10.0, 12.0], [20.0, 12.0]]}
        ],
    }
    assert sent[0]["headers"]["Content-Type"] == "application/json"
    assert [(t.task, t.model, t.requests, t.failed) for t in trace.tallies] == [("face_group", FACES_MODEL_ALIAS, 1, 0)]


def test_group_label_count_mismatch_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reply with fewer labels than tracks is a failure."""
    monkeypatch.setenv("FACES_API_BASE", "http://router:4000")
    monkeypatch.setattr(faces.httpx, "post", lambda url, **kwargs: _reply(url, {"labels": [0]}))

    with trace_inference() as trace:
        assert group_faces([_track((0,)), _track((1,))]) is None
    assert trace.tallies[0].failed == 1


@pytest.mark.parametrize("payload", [{"labels": ["a", 0]}, {"labels": [True, 0]}, {"nope": []}, [0, 1]])
def test_group_malformed_reply_is_none(monkeypatch: pytest.MonkeyPatch, payload: object) -> None:
    """Labels that are not integers, or no labels at all, are a failure.

    Args:
        monkeypatch: Pytest patcher.
        payload: The malformed reply.
    """
    monkeypatch.setenv("FACES_API_BASE", "http://router:4000")
    monkeypatch.setattr(faces.httpx, "post", lambda url, **kwargs: _reply(url, payload))

    assert group_faces([_track((0,)), _track((1,))]) is None


def test_group_over_budget_sends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A body over MAX_REQUEST_BYTES is not sent at all."""
    monkeypatch.setenv("FACES_API_BASE", "http://router:4000")
    monkeypatch.setattr(faces.httpx, "post", _no_request)
    big = b"\xff" * (MAX_REQUEST_BYTES // 2)

    with trace_inference() as trace:
        assert group_faces([_track((0,), jpeg=big), _track((1,), jpeg=big)]) is None
    assert trace.tallies == []


def test_group_http_error_is_none(monkeypatch: pytest.MonkeyPatch, log_lines: list[str]) -> None:
    """A failed group request degrades to None, logged by class and status."""
    monkeypatch.setenv("FACES_API_BASE", "http://router:4000")
    monkeypatch.setattr(faces.httpx, "post", lambda url, **kwargs: _reply(url, {"detail": "busy"}, status=503))

    assert group_faces([_track((0,))]) is None
    assert any("HTTPStatusError (HTTP 503)" in line for line in log_lines)
