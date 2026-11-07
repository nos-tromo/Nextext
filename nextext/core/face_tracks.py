"""Face tracks, shot cuts and crop quality — pure geometry, no model.

The faces service finds faces frame by frame; this module links the detections
of consecutive samples into tracks (the same face over time) and scores how
good each crop is. Everything here is arithmetic on boxes, landmarks and grey
levels with NumPy and Pillow: nothing recognizes anyone.

A detection continues a track when its box overlaps the track's last box by
an IoU of at least :data:`IOU_MIN`, its area is within :data:`MAX_AREA_RATIO`
of it, and at most one sample went by without the face. A shot cut — a large
mean grey-level change between tiny thumbnails of consecutive samples — closes
every track, so a face in the same spot of the next shot starts a new one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

__all__ = [
    "CUT_THRESHOLD",
    "IOU_MIN",
    "MAX_AREA_RATIO",
    "MAX_SAMPLE_GAP",
    "Box",
    "Detection",
    "Track",
    "assign_tracks",
    "frontality",
    "grey_thumb",
    "iou",
    "is_cut",
    "merge_spans",
    "quality",
    "sharpness",
]

# (x, y, width, height) in source pixels.
Box = tuple[int, int, int, int]
Point = tuple[float, float]

IOU_MIN = 0.5
# Largest area change between a track's last box and a detection continuing it.
MAX_AREA_RATIO = 1.5
# A track continues across at most one sample without its face.
MAX_SAMPLE_GAP = 2
# Mean absolute grey-level change (0-255) between thumbnails that marks a cut.
CUT_THRESHOLD = 30.0
THUMB_SIZE = (32, 18)
# Faces this many pixels or more on their short side count as full size.
_FULL_SIZE_PX = 112
# Laplacian variance at which a crop counts as fully sharp.
_FULL_SHARPNESS_VARIANCE = 100.0


@dataclass(frozen=True)
class Detection:
    """One face found in one sample, in source pixels.

    Attributes:
        sample: Index of the sample it was found in.
        time_sec: That sample's time in seconds.
        box: The face's box.
        score: The detector's confidence.
        landmarks: The five landmark points (right eye, left eye, nose tip,
            right and left mouth corner).
    """

    sample: int
    time_sec: float
    box: Box
    score: float
    landmarks: tuple[Point, ...]


@dataclass
class Track:
    """One face followed over consecutive samples.

    Attributes:
        id: Track number, unique within one video, counting from ``0``.
        detections: The track's detections, in sample order.
        closed: Whether the track can no longer be continued (a shot cut, or
            more than one sample without the face).
    """

    id: int
    detections: list[Detection] = field(default_factory=list)
    closed: bool = False

    @property
    def first_time(self) -> float:
        """Time of the track's first detection.

        Returns:
            float: Seconds.
        """
        return self.detections[0].time_sec

    @property
    def last_time(self) -> float:
        """Time of the track's last detection.

        Returns:
            float: Seconds.
        """
        return self.detections[-1].time_sec

    @property
    def last_sample(self) -> int:
        """Sample index of the track's last detection.

        Returns:
            int: The index.
        """
        return self.detections[-1].sample

    @property
    def mean_area(self) -> float:
        """Mean box area over the track's detections.

        Returns:
            float: Square pixels.
        """
        return sum(d.box[2] * d.box[3] for d in self.detections) / len(self.detections)


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes.

    Args:
        a: The first box.
        b: The second box.

    Returns:
        float: In ``[0, 1]``; ``0.0`` when either box is empty.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    overlap_w = min(ax + aw, bx + bw) - max(ax, bx)
    overlap_h = min(ay + ah, by + bh) - max(ay, by)
    if overlap_w <= 0 or overlap_h <= 0:
        return 0.0
    intersection = overlap_w * overlap_h
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _area_ratio(a: Box, b: Box) -> float:
    """Ratio of the larger box area to the smaller.

    Args:
        a: The first box.
        b: The second box.

    Returns:
        float: ``>= 1``; infinite when one box is empty.
    """
    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    smaller, larger = sorted((area_a, area_b))
    return larger / smaller if smaller > 0 else math.inf


def grey_thumb(image: Image.Image) -> np.ndarray:
    """Shrink a frame to the tiny grey thumbnail the cut detector compares.

    Args:
        image: The frame.

    Returns:
        np.ndarray: ``18 x 32`` grey levels as ``float32``.
    """
    return np.asarray(image.convert("L").resize(THUMB_SIZE, Image.Resampling.BILINEAR), dtype=np.float32)


def is_cut(prev: np.ndarray | None, cur: np.ndarray) -> bool:
    """Report whether two consecutive samples belong to different shots.

    Args:
        prev: The previous sample's thumbnail, or ``None`` for the first sample.
        cur: This sample's thumbnail.

    Returns:
        bool: ``True`` when the mean absolute grey-level change exceeds
            :data:`CUT_THRESHOLD`.
    """
    if prev is None or prev.shape != cur.shape:
        return False
    return float(np.mean(np.abs(cur - prev))) > CUT_THRESHOLD


def assign_tracks(tracks: list[Track], detections: Sequence[Detection], *, sample: int, cut: bool) -> list[Track]:
    """Link one sample's detections to open tracks, opening new tracks for the rest.

    Call it once per sample, in sample order, also for a sample without
    detections, so tracks that ran out of samples close. On a cut every open
    track closes first. A detection may continue an open track whose last
    sample is at most :data:`MAX_SAMPLE_GAP` back, whose last box it overlaps
    by at least :data:`IOU_MIN` and whose area it matches within
    :data:`MAX_AREA_RATIO`; pairs are taken greedily by descending IoU, and a
    track takes at most one detection per sample.

    Args:
        tracks: Every track so far; extended and updated in place.
        detections: This sample's detections.
        sample: This sample's index.
        cut: Whether this sample opens a new shot.

    Returns:
        list[Track]: The track each detection now belongs to, in the order of
            ``detections``.
    """
    for track in tracks:
        if not track.closed and (cut or track.last_sample < sample - MAX_SAMPLE_GAP):
            track.closed = True
    open_tracks = [track for track in tracks if not track.closed]
    pairs: list[tuple[float, int, int]] = []
    for track_index, track in enumerate(open_tracks):
        last_box = track.detections[-1].box
        for detection_index, detection in enumerate(detections):
            overlap = iou(last_box, detection.box)
            if overlap >= IOU_MIN and _area_ratio(last_box, detection.box) <= MAX_AREA_RATIO:
                pairs.append((overlap, track_index, detection_index))
    pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))
    owners: list[Track | None] = [None] * len(detections)
    taken: set[int] = set()
    for _overlap, track_index, detection_index in pairs:
        if track_index in taken or owners[detection_index] is not None:
            continue
        taken.add(track_index)
        owners[detection_index] = open_tracks[track_index]
    next_id = max((track.id for track in tracks), default=-1) + 1
    result: list[Track] = []
    for detection, owner in zip(detections, owners, strict=True):
        if owner is None:
            owner = Track(id=next_id)
            next_id += 1
            tracks.append(owner)
        owner.detections.append(detection)
        result.append(owner)
    return result


def frontality(landmarks: Sequence[Point]) -> float:
    """Estimate how squarely a face looks at the camera from its landmarks.

    Args:
        landmarks: The face's landmarks; the first two are the eyes, the third
            the nose tip.

    Returns:
        float: ``1.0`` for a nose centred between the eyes, falling to ``0.0``
            as it reaches half the eye distance off centre; ``0.0`` without
            usable landmarks.
    """
    if len(landmarks) < 3:
        return 0.0
    (eye0_x, eye0_y), (eye1_x, eye1_y), (nose_x, _nose_y) = landmarks[0], landmarks[1], landmarks[2]
    eye_dist = math.hypot(eye1_x - eye0_x, eye1_y - eye0_y)
    if eye_dist == 0:
        return 0.0
    return max(0.0, 1.0 - abs(nose_x - (eye0_x + eye1_x) / 2) / (0.5 * eye_dist))


def sharpness(grey: np.ndarray) -> float:
    """Measure a crop's sharpness by the variance of its Laplacian.

    Args:
        grey: The crop's grey levels (0-255), two-dimensional.

    Returns:
        float: In ``[0, 1]``; ``0.0`` for a flat or too-small patch.
    """
    if grey.ndim != 2 or grey.shape[0] < 3 or grey.shape[1] < 3:
        return 0.0
    values = grey.astype(np.float64)
    laplacian = values[1:-1, :-2] + values[1:-1, 2:] + values[:-2, 1:-1] + values[2:, 1:-1] - 4.0 * values[1:-1, 1:-1]
    return min(1.0, float(laplacian.var()) / _FULL_SHARPNESS_VARIANCE)


def quality(score: float, short_side: int, frontal: float, sharp: float) -> float:
    """Score how good a crop is to show and to pass on.

    Args:
        score: The detector's confidence.
        short_side: The face box's shorter side in source pixels.
        frontal: :func:`frontality` of the face.
        sharp: :func:`sharpness` of the crop.

    Returns:
        float: Higher is better; at most ``score``.
    """
    return score * min(1.0, short_side / _FULL_SIZE_PX) * (0.5 + 0.5 * frontal) * (0.5 + 0.5 * sharp)


def merge_spans(spans: Sequence[tuple[float, float]], *, gap: float) -> list[tuple[float, float]]:
    """Merge time spans that overlap or lie at most ``gap`` apart.

    Args:
        spans: ``(start, end)`` pairs in seconds, in any order.
        gap: Largest distance between two spans that still joins them.

    Returns:
        list[tuple[float, float]]: The merged spans, by start time.
    """
    merged: list[tuple[float, float]] = []
    for start, end in sorted(spans):
        if merged and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
