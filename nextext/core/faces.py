"""Faces agent: HTTP client for the out-of-process ``/faces/detect`` and ``/faces/group`` service.

Nextext runs no face model. Detection and grouping run behind the inference
router (e.g. ``nos-tromo/vllm-service``), and the features that drive grouping
never leave that service: ``/faces/group`` takes face crops and returns one
label per track, nothing else.

The endpoint is resolved by :func:`nextext.utils.env_cfg.load_faces_env`:
``FACES_API_BASE`` when set, the central ``OPENAI_API_BASE`` (one trailing
``/v1`` stripped) by default, or empty when ``FACES_API_BASE`` is an explicit off
token. Both calls are **fail-soft**: no endpoint, a transport or HTTP error, or a
reply that breaks the contract return ``None``, so the face stage degrades
instead of failing the job. A ``503`` whose ``Retry-After`` asks for at most
:data:`RETRY_AFTER_MAX_S` seconds is waited out and retried once. Errors are logged by class and status only — never
a response body, never an image.

Request/response contract::

    POST {FACES_API_BASE}/faces/detect
        application/json:
            {"images": ["<base64 JPEG/PNG>", ... at most 8], "min_score": 0.7}
        Authorization: Bearer <OPENAI_API_KEY>   (optional)

    200 OK
        {"results": [
            {"faces": [{"box": [x, y, w, h], "score": s,
                        "landmarks": [[x, y], ... 5]}]},
            {"error": "undecodable"}
        ]}
        One result per image, in order; coordinates normalized to [0, 1] of
        the submitted image; landmarks in YuNet order (right eye, left eye,
        nose tip, right mouth corner, left mouth corner).

    POST {FACES_API_BASE}/faces/group
        application/json:
            {"tracks": [{"samples": [3, 4, 5],
                         "faces": [{"image": "<base64 JPEG>",
                                    "landmarks": [[x, y], ... crop pixels]}]}]}
        Tracks that share a sample index were on screen together and are
        never merged.

    200 OK
        {"labels": [0, 0, 1, -1]}
        One label per track, numbered by each cluster's first track; ``-1``
        for a track with no usable face.
"""

from __future__ import annotations

import base64
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx as httpx  # explicit re-export so tests can monkeypatch faces.httpx
from loguru import logger

from nextext.core.inference_trace import correlation_headers, record_inference
from nextext.utils.env_cfg import FacesConfig, load_faces_env
from nextext.utils.log_cfg import error_label

__all__ = [
    "FACES_MODEL_ALIAS",
    "MAX_REQUEST_BYTES",
    "DetectedFace",
    "GroupFace",
    "GroupTrack",
    "detect_faces",
    "group_faces",
    "group_request_body",
]

# The faces service takes no model field; the audit log names it by its endpoint.
FACES_MODEL_ALIAS = "faces"
# Largest request body sent; the service itself refuses bodies over 32 MiB.
MAX_REQUEST_BYTES = 16 * 2**20
# A busy service (503) that asks to wait at most this long gets one retry.
RETRY_AFTER_MAX_S = 10.0
# What _post returns when no usable reply came back.
_FAILED = object()

Point = tuple[float, float]


@dataclass(frozen=True)
class DetectedFace:
    """One face found in one image, in coordinates normalized to that image.

    Attributes:
        box: ``(x, y, width, height)``, each in ``[0, 1]`` of the image's size.
        score: The detector's confidence.
        landmarks: The five YuNet points (right eye, left eye, nose tip, right
            and left mouth corner), normalized like ``box``.
    """

    box: tuple[float, float, float, float]
    score: float
    landmarks: tuple[Point, ...]


@dataclass(frozen=True)
class GroupFace:
    """One face crop sent for grouping.

    Attributes:
        jpeg: The crop, JPEG-encoded.
        landmarks: The face's landmarks in the crop's own pixels.
    """

    jpeg: bytes
    landmarks: tuple[Point, ...]


@dataclass(frozen=True)
class GroupTrack:
    """One track — the same face over consecutive samples — sent for grouping.

    Attributes:
        samples: The sample indices the track spans; tracks sharing one were
            on screen together, so the service keeps them apart.
        faces: A few good crops of the face.
    """

    samples: tuple[int, ...]
    faces: tuple[GroupFace, ...]


def _headers(config: FacesConfig) -> dict[str, str]:
    """Build the request headers: correlation id and bearer token.

    Args:
        config: The resolved faces configuration.

    Returns:
        dict[str, str]: The headers.
    """
    headers: dict[str, str] = correlation_headers()
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    return headers


def _retry_after(response: httpx.Response) -> float | None:
    """Read how long a busy service asks to wait, when it is short enough to wait.

    Args:
        response: The reply.

    Returns:
        float | None: Seconds for a ``503`` whose ``Retry-After`` is a number
            of seconds up to :data:`RETRY_AFTER_MAX_S`; ``None`` otherwise.
    """
    if response.status_code != 503:
        return None
    try:
        seconds = float(response.headers.get("Retry-After", "").strip())
    except ValueError:
        return None
    if not math.isfinite(seconds) or not 0 <= seconds <= RETRY_AFTER_MAX_S:
        return None
    return seconds


def _post(url: str, *, task: str, label: str, config: FacesConfig, **request: Any) -> object:
    """POST to the faces service, retrying once when it is briefly busy.

    Every request sent is recorded under ``task``: a failed one here, a
    successful one by the caller once it has checked the reply.

    Args:
        url: The endpoint.
        task: The inference task the request is recorded under.
        label: How the request is named in the logs.
        config: The faces configuration (timeout, bearer token).
        **request: ``json=`` or ``content=`` and ``headers=`` for ``httpx.post``.

    Returns:
        object: The decoded JSON reply, or :data:`_FAILED`.
    """
    headers = {**_headers(config), **request.pop("headers", {})}
    for attempt in range(2):
        try:
            response = httpx.post(url, headers=headers, timeout=config.timeout, **request)
            wait = _retry_after(response) if attempt == 0 else None
            if wait is not None:
                record_inference(task, FACES_MODEL_ALIAS, ok=False)
                logger.warning("{} at {} is busy (HTTP 503); retrying once in {}s.", label, url, wait)
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            record_inference(task, FACES_MODEL_ALIAS, ok=False)
            logger.warning("{} request to {} failed: {}.", label, url, error_label(exc))
            return _FAILED
    return _FAILED


def _b64(data: bytes) -> str:
    """Encode bytes as base64 text for a JSON body.

    Args:
        data: The bytes.

    Returns:
        str: ASCII base64.
    """
    return base64.b64encode(data).decode("ascii")


def _number(value: Any) -> float | None:
    """Read a finite JSON number.

    Args:
        value: A decoded JSON value.

    Returns:
        float | None: The number, or ``None`` when it is not a finite number.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _points(value: Any) -> tuple[Point, ...] | None:
    """Read a list of ``[x, y]`` pairs.

    Args:
        value: A decoded JSON value.

    Returns:
        tuple[Point, ...] | None: The points, or ``None`` when malformed.
    """
    if not isinstance(value, list):
        return None
    points: list[Point] = []
    for pair in value:
        if not isinstance(pair, list) or len(pair) != 2:
            return None
        x, y = _number(pair[0]), _number(pair[1])
        if x is None or y is None:
            return None
        points.append((x, y))
    return tuple(points)


def _parse_face(value: Any) -> DetectedFace | None:
    """Read one face of a detect reply.

    Args:
        value: A decoded JSON value.

    Returns:
        DetectedFace | None: The face, or ``None`` when malformed.
    """
    if not isinstance(value, dict):
        return None
    raw_box = value.get("box")
    if not isinstance(raw_box, list) or len(raw_box) != 4:
        return None
    box = [_number(part) for part in raw_box]
    score = _number(value.get("score"))
    landmarks = _points(value.get("landmarks", []))
    if score is None or landmarks is None:
        return None
    x, y, width, height = box
    if x is None or y is None or width is None or height is None:
        return None
    return DetectedFace(box=(x, y, width, height), score=score, landmarks=landmarks)


def _parse_results(payload: Any, expected: int) -> list[list[DetectedFace] | None] | None:
    """Read a detect reply against the contract.

    Args:
        payload: The decoded JSON body.
        expected: How many images were sent.

    Returns:
        list[list[DetectedFace] | None] | None: One entry per image (``None``
            for an image the service could not decode), or ``None`` when the
            reply is malformed.
    """
    if not isinstance(payload, dict):
        return None
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != expected:
        return None
    parsed: list[list[DetectedFace] | None] = []
    for result in results:
        if not isinstance(result, dict):
            return None
        found = result.get("faces")
        if found is None and "error" in result:
            parsed.append(None)
            continue
        if not isinstance(found, list):
            return None
        image_faces: list[DetectedFace] = []
        for raw_face in found:
            face = _parse_face(raw_face)
            if face is None:
                return None
            image_faces.append(face)
        parsed.append(image_faces)
    return parsed


def detect_faces(
    images: Sequence[bytes], *, min_score: float = 0.7, config: FacesConfig | None = None
) -> list[list[DetectedFace] | None] | None:
    """Find the faces in a batch of images via ``/faces/detect``.

    One request per call, recorded once as ``face_detect``.

    Args:
        images: Encoded images (JPEG or PNG), at most eight.
        min_score: Lowest detector confidence the service should return.
        config: The faces configuration, when the caller read it already (one
            video's batches share one); read from the environment otherwise.

    Returns:
        list[list[DetectedFace] | None] | None: The faces of each image, in
            order, with ``None`` for an image the service could not decode;
            ``None`` overall when no endpoint is configured or the request
            failed or broke the contract.
    """
    config = config or load_faces_env()
    if not config.api_base:
        return None
    if not images:
        return []
    url = f"{config.api_base}/faces/detect"
    payload = _post(
        url,
        task="face_detect",
        label="Face detection",
        config=config,
        json={"images": [_b64(image) for image in images], "min_score": min_score},
    )
    if payload is _FAILED:
        return None
    parsed = _parse_results(payload, len(images))
    record_inference("face_detect", FACES_MODEL_ALIAS, ok=parsed is not None)
    if parsed is None:
        logger.warning("Face detection reply from {} did not match the contract.", url)
    return parsed


def _parse_labels(payload: Any, expected: int) -> list[int] | None:
    """Read a group reply against the contract.

    Args:
        payload: The decoded JSON body.
        expected: How many tracks were sent.

    Returns:
        list[int] | None: One label per track, or ``None`` when malformed.
    """
    if not isinstance(payload, dict):
        return None
    labels = payload.get("labels")
    if not isinstance(labels, list) or len(labels) != expected:
        return None
    if not all(isinstance(label, int) and not isinstance(label, bool) and label >= -1 for label in labels):
        return None
    return list(labels)


def group_request_body(tracks: Sequence[GroupTrack]) -> bytes:
    """Serialize a ``/faces/group`` request, so its size can be checked before sending.

    Args:
        tracks: The tracks, each with its sample indices and a few crops.

    Returns:
        bytes: The compact JSON body.
    """
    return json.dumps(
        {
            "tracks": [
                {
                    "samples": list(track.samples),
                    "faces": [
                        {"image": _b64(face.jpeg), "landmarks": [list(point) for point in face.landmarks]}
                        for face in track.faces
                    ],
                }
                for track in tracks
            ]
        },
        separators=(",", ":"),
    ).encode("ascii")


def group_faces(tracks: Sequence[GroupTrack], *, config: FacesConfig | None = None) -> list[int] | None:
    """Group tracks of the same person via ``/faces/group``.

    The service returns labels only; the features it groups by never reach
    Nextext. One request per call, recorded once as ``face_group``. A body
    over :data:`MAX_REQUEST_BYTES` is not sent at all.

    Args:
        tracks: The tracks, each with its sample indices and a few crops.
        config: The faces configuration, when the caller read it already;
            read from the environment otherwise.

    Returns:
        list[int] | None: One label per track (``-1`` for a track the service
            could not use), or ``None`` when no endpoint is configured, the
            body is over budget, or the request failed or broke the contract.
    """
    config = config or load_faces_env()
    if not config.api_base:
        return None
    if not tracks:
        return []
    body = group_request_body(tracks)
    if len(body) > MAX_REQUEST_BYTES:
        logger.warning(
            "Face grouping skipped: the request would be {} bytes, over the {} byte budget.",
            len(body),
            MAX_REQUEST_BYTES,
        )
        return None
    url = f"{config.api_base}/faces/group"
    payload = _post(
        url,
        task="face_group",
        label="Face grouping",
        config=config,
        content=body,
        headers={"Content-Type": "application/json"},
    )
    if payload is _FAILED:
        return None
    labels = _parse_labels(payload, len(tracks))
    record_inference("face_group", FACES_MODEL_ALIAS, ok=labels is not None)
    if labels is None:
        logger.warning("Face grouping reply from {} did not match the contract.", url)
    return labels
