"""Face extraction: find, track and (optionally) group the faces of a video.

The video is sampled by time (:func:`nextext.core.keyframes.sample_frames`), and
the samples go to the faces service in small batches, each image scaled down
for detection. The boxes that come back are mapped onto the source frame,
filtered, linked into tracks (:mod:`nextext.core.face_tracks`) and cut out of
the source while it is still in memory: per track the best shot with a wide
margin, a few smaller crops, a thumbnail, and — in ``group`` mode — up to three
crops for ``/faces/group``, good ones first. Only the batch's source frames are held at
once.

After the last batch the tracks are capped, grouped per person when the mode
and the service allow it, merged into entries, capped again, and numbered by
first appearance. No model runs here, and nothing biometric comes back: the
service returns boxes and labels only.

Fail-soft: a failed batch leaves its samples without faces (``partial``), no
answered batch at all gives ``unavailable``, and a failed grouping leaves every
track its own entry.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger
from PIL import Image

from nextext.core import faces as faces_api
from nextext.core.face_tracks import (
    Box,
    Detection,
    Track,
    assign_tracks,
    frontality,
    grey_thumb,
    is_cut,
    merge_spans,
    quality,
    sharpness,
)
from nextext.core.faces import DetectedFace, GroupFace, GroupTrack, detect_faces, group_faces
from nextext.core.keyframes import SampledFrame, sample_frames
from nextext.utils.env_cfg import FacesConfig, load_faces_env

__all__ = [
    "DetectionRow",
    "FaceCrop",
    "FaceEntry",
    "FacesOutcome",
    "FacesResult",
    "extract_faces",
]

FacesOutcome = Literal["ok", "partial", "unavailable"]
Point = tuple[float, float]

# Detection requests.
BATCH_IMAGES = 8
DETECT_MAX_SIDE = 1280
DETECT_JPEG_QUALITY = 85
# Source frames held at once (RGB bytes), so 4K video batches fewer frames.
SOURCE_BATCH_BYTES = 100 * 2**20
# Filtering and caps.
MAX_FACES_PER_FRAME = 20
MIN_FACE_PX = 40
MAX_TRACKS = 500
MAX_ENTRIES = 100
# Detect batches failing in a row after which the service counts as down: the
# rest of the video is neither decoded nor sent, so a hung service cannot hold
# a job for hours (one timeout per batch).
FAILED_BATCHES_TO_STOP = 3
# A face good enough to found a person on: a track with one is an anchor; a
# track without one is sent as a weak track, which may only join a person.
GOOD_SCORE = 0.85
GOOD_MIN_PX = 64
GOOD_FRONTALITY = 0.5
GROUP_FACES_PER_TRACK = 3
# Crops kept per entry besides the best shot.
EXTRA_CROPS = 3
# Grey patch size the sharpness of a face is measured on, so faces of any size compare.
_SHARPNESS_PATCH = (64, 64)


@dataclass(frozen=True)
class _CropSpec:
    """How one kind of crop is cut.

    Attributes:
        margin: Extra room around the box, as a fraction of its size per side.
        max_side: Largest width or height of the encoded crop.
        jpeg_quality: JPEG quality.
    """

    margin: float
    max_side: int
    jpeg_quality: int


BEST_CROP = _CropSpec(margin=0.5, max_side=1024, jpeg_quality=90)
SMALL_CROP = _CropSpec(margin=0.25, max_side=256, jpeg_quality=85)
GROUP_CROP = _CropSpec(margin=0.25, max_side=160, jpeg_quality=90)
THUMBNAIL = _CropSpec(margin=0.15, max_side=128, jpeg_quality=80)


@dataclass(frozen=True)
class FaceCrop:
    """One face cut from a source frame.

    Attributes:
        time_sec: When the frame was shown, in seconds.
        box: The face's box in source pixels (the crop itself has a margin).
        score: The detector's confidence.
        quality: The crop's quality (:func:`nextext.core.face_tracks.quality`).
        jpeg: The crop, JPEG-encoded, without metadata.
    """

    time_sec: float
    box: Box
    score: float
    quality: float
    jpeg: bytes


@dataclass(frozen=True)
class FaceEntry:
    """One person (grouped) or one face track (ungrouped) of a video.

    Attributes:
        number: Position by first appearance, from ``1``.
        assigned: Whether the faces service grouped it as a person; otherwise
            it is a single track.
        appearances: ``(start, end)`` spans in seconds when it was on screen.
        detections: How many sampled frames it was found in.
        best: Its best shot, with a wide margin, at most 1024 px.
        crops: Up to three more crops, at most 256 px, best first.
        thumbnail: A small JPEG of the best shot, at most 128 px.
    """

    number: int
    assigned: bool
    appearances: tuple[tuple[float, float], ...]
    detections: int
    best: FaceCrop
    crops: tuple[FaceCrop, ...]
    thumbnail: bytes


@dataclass(frozen=True)
class DetectionRow:
    """One detection of a kept entry, for the face table.

    Attributes:
        time_sec: The sample's time in seconds.
        entry: The entry's number.
        box: The face's box in source pixels.
        score: The detector's confidence.
    """

    time_sec: float
    entry: int
    box: Box
    score: float


@dataclass(frozen=True)
class FacesResult:
    """The faces of one video.

    Attributes:
        outcome: ``ok`` when every batch was answered, ``partial`` when some
            failed or held undecodable images, ``unavailable`` when none was
            answered (and so no entries).
        grouped: Whether the entries were grouped per person by the service.
        truncated: Whether faces, tracks or entries were dropped by a cap.
        entries: The entries, by number.
        rows: Every detection of the kept entries, by time.
        frames_sent: Frames sent for detection — what the stage judged.
        tracks_sent: Tracks sent for grouping.
    """

    outcome: FacesOutcome
    grouped: bool
    truncated: bool
    entries: tuple[FaceEntry, ...]
    rows: tuple[DetectionRow, ...]
    frames_sent: int
    tracks_sent: int


@dataclass
class _TrackCrops:
    """What is kept of one track while the video is read.

    Attributes:
        best: The highest-quality shot so far.
        thumbnail: The thumbnail of that shot.
        small: The best few smaller crops, best first.
        group: The best few crops for grouping, good ones first, then by
            quality, each with that ranking.
    """

    best: FaceCrop | None = None
    thumbnail: bytes = b""
    small: list[FaceCrop] = field(default_factory=list)
    group: list[tuple[tuple[bool, float], GroupFace]] = field(default_factory=list)

    def group_faces(self) -> tuple[bool, list[GroupFace]]:
        """The crops to group this track by, and whether it is an anchor.

        Returns:
            tuple[bool, list[GroupFace]]: ``True`` and only the good crops when
                there is one; else ``False`` and the best crops there are.
        """
        good = [face for (is_good, _quality), face in self.group if is_good]
        if good:
            return True, good
        return False, [face for _rank, face in self.group]


def _encode_jpeg(image: Image.Image, jpeg_quality: int) -> bytes:
    """Encode an image as a JPEG without metadata.

    Args:
        image: The image.
        jpeg_quality: JPEG quality.

    Returns:
        bytes: The JPEG.
    """
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=jpeg_quality)
    return buffer.getvalue()


def _scaled(image: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    """Scale an image down so its long side is at most ``max_side``.

    Args:
        image: The image.
        max_side: Largest width or height.

    Returns:
        tuple[Image.Image, float]: The (possibly) scaled image and the scale.
    """
    scale = min(1.0, max_side / max(image.size))
    if scale >= 1.0:
        return image, 1.0
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS), scale


def _cut(image: Image.Image, box: Box, spec: _CropSpec) -> tuple[bytes, float, float, float]:
    """Cut a box with a margin out of a frame, clamped to the frame.

    Args:
        image: The source frame.
        box: The face's box in source pixels.
        spec: Margin, size and quality of the crop.

    Returns:
        tuple[bytes, float, float, float]: The JPEG, the crop's left and top
            edge in source pixels, and the scale applied to it.
    """
    x, y, width, height = box
    left = max(0, math.floor(x - spec.margin * width))
    top = max(0, math.floor(y - spec.margin * height))
    right = min(image.width, math.ceil(x + width + spec.margin * width))
    bottom = min(image.height, math.ceil(y + height + spec.margin * height))
    crop, scale = _scaled(image.crop((left, top, right, bottom)), spec.max_side)
    return _encode_jpeg(crop, spec.jpeg_quality), float(left), float(top), scale


def _source_bytes(frame: SampledFrame) -> int:
    """Memory a source frame holds as RGB.

    Args:
        frame: The sample.

    Returns:
        int: Bytes.
    """
    return frame.image.width * frame.image.height * 3


def _to_source(box: Box, scale: float) -> Box:
    """Map a box from a sampled image back to the source frame's own pixels.

    Args:
        box: The box in the sampled image.
        scale: The image's size over the source's.

    Returns:
        Box: The box in source pixels.
    """
    if scale >= 1.0 or scale <= 0:
        return box
    x, y, width, height = box
    return (round(x / scale), round(y / scale), round(width / scale), round(height / scale))


def _clamp(value: int, upper: int) -> int:
    """Clamp a coordinate to ``[0, upper]``.

    Args:
        value: The coordinate.
        upper: The frame's extent.

    Returns:
        int: The clamped coordinate.
    """
    return max(0, min(upper, value))


class _Extraction:
    """The running state of one video's extraction."""

    def __init__(self, mode: Literal["detect", "group"], interval_s: float, config: FacesConfig) -> None:
        """Start an empty extraction.

        Args:
            mode: ``detect`` or ``group``.
            interval_s: The requested sampling interval.
            config: The faces service configuration, read once for the video.
        """
        self.mode = mode
        self.interval_s = interval_s
        self.config = config
        self.tracks: list[Track] = []
        self.open_tracks: list[Track] = []
        self.crops: dict[int, _TrackCrops] = {}
        self.sample_times: list[float] = []
        # Sample -> how far the sampler scaled that frame down, for reporting
        # boxes in source pixels.
        self.scales: dict[int, float] = {}
        # The clip's duration as the sampler measured it; 0.0 when unknown.
        self.duration_sec = 0.0
        self.prev_thumb: np.ndarray | None = None
        self.frames_sent = 0
        self.batches_answered = 0
        self.batches_failed = 0
        self.failed_in_a_row = 0
        # The interval tracks are ranked by, fixed at its first use so that
        # pruning while reading and the final cap rank alike.
        self._ranking_interval: float | None = None
        self.undecodable = 0
        self.truncated = False

    def process(self, batch: Sequence[SampledFrame]) -> None:
        """Detect the faces of one batch, track them and keep their crops.

        Args:
            batch: Consecutive samples.
        """
        images = [_encode_jpeg(_scaled(frame.image, DETECT_MAX_SIDE)[0], DETECT_JPEG_QUALITY) for frame in batch]
        self.frames_sent += len(batch)
        results = detect_faces(images, config=self.config)
        if results is None:
            self.batches_failed += 1
            self.failed_in_a_row += 1
            if self.stopped:
                logger.warning(
                    "Face detection failed for {} batches in a row; the rest of the video is not sampled.",
                    self.failed_in_a_row,
                )
        else:
            self.batches_answered += 1
            self.failed_in_a_row = 0
        for position, frame in enumerate(batch):
            self.sample_times.append(frame.time_sec)
            self.scales[frame.index] = frame.scale
            self.duration_sec = max(self.duration_sec, frame.clip_duration_sec)
            thumb = grey_thumb(frame.image)
            cut = is_cut(self.prev_thumb, thumb)
            self.prev_thumb = thumb
            found: list[DetectedFace] = []
            if results is not None:
                image_faces = results[position]
                if image_faces is None:
                    self.undecodable += 1
                else:
                    found = image_faces
            detections = self._detections(frame, found)
            owners = assign_tracks(self.tracks, detections, sample=frame.index, cut=cut, open_tracks=self.open_tracks)
            for detection, track in zip(detections, owners, strict=True):
                self._keep(frame, detection, track)
        self._prune_closed()

    def _detections(self, frame: SampledFrame, found: Sequence[DetectedFace]) -> list[Detection]:
        """Map one sample's faces to source pixels and filter them.

        Args:
            frame: The sample.
            found: The faces the service reported, normalized to the sent image.

        Returns:
            list[Detection]: At most :data:`MAX_FACES_PER_FRAME` faces of at
                least :data:`MIN_FACE_PX`, most prominent (score times area) first.
        """
        width, height = frame.image.size
        detections: list[Detection] = []
        for face in found:
            nx, ny, nw, nh = face.box
            left = _clamp(round(nx * width), width)
            top = _clamp(round(ny * height), height)
            right = _clamp(round((nx + nw) * width), width)
            bottom = _clamp(round((ny + nh) * height), height)
            box: Box = (left, top, right - left, bottom - top)
            if min(box[2], box[3]) < MIN_FACE_PX:
                continue
            landmarks = tuple((px * width, py * height) for px, py in face.landmarks)
            detections.append(
                Detection(sample=frame.index, time_sec=frame.time_sec, box=box, score=face.score, landmarks=landmarks)
            )
        detections.sort(key=lambda detection: detection.score * detection.box[2] * detection.box[3], reverse=True)
        if len(detections) > MAX_FACES_PER_FRAME:
            self.truncated = True
            del detections[MAX_FACES_PER_FRAME:]
        return detections

    def _keep(self, frame: SampledFrame, detection: Detection, track: Track) -> None:
        """Score one detection's crop and keep what its track needs of it.

        Crops are cut from the sampled image; the boxes they carry are mapped
        back to the source's own pixels.

        Args:
            frame: The sample the detection was found in.
            detection: The detection.
            track: The track it belongs to.
        """
        image = frame.image
        source_box = _to_source(detection.box, frame.scale)
        x, y, width, height = detection.box
        patch = image.crop((x, y, x + width, y + height)).convert("L").resize(_SHARPNESS_PATCH)
        frontal = frontality(detection.landmarks)
        score = quality(detection.score, min(width, height), frontal, sharpness(np.asarray(patch, dtype=np.float32)))
        kept = self.crops.setdefault(track.id, _TrackCrops())
        if kept.best is None or score > kept.best.quality:
            best_jpeg = _cut(image, detection.box, BEST_CROP)[0]
            kept.best = FaceCrop(detection.time_sec, source_box, detection.score, score, best_jpeg)
            kept.thumbnail = _cut(image, detection.box, THUMBNAIL)[0]
        if len(kept.small) <= EXTRA_CROPS or score > kept.small[-1].quality:
            small_jpeg = _cut(image, detection.box, SMALL_CROP)[0]
            kept.small.append(FaceCrop(detection.time_sec, source_box, detection.score, score, small_jpeg))
            kept.small.sort(key=lambda crop: crop.quality, reverse=True)
            del kept.small[EXTRA_CROPS + 1 :]
        if self.mode == "group":
            rank = (self._is_good(detection, frontal), score)
            if len(kept.group) < GROUP_FACES_PER_TRACK or rank > kept.group[-1][0]:
                group_jpeg, left, top, scale = _cut(image, detection.box, GROUP_CROP)
                landmarks = tuple(((px - left) * scale, (py - top) * scale) for px, py in detection.landmarks)
                kept.group.append((rank, GroupFace(jpeg=group_jpeg, landmarks=landmarks)))
                kept.group.sort(key=lambda item: item[0], reverse=True)
                del kept.group[GROUP_FACES_PER_TRACK:]

    def tracks_with_crops(self) -> list[Track]:
        """The tracks whose crops are still kept, without walking every track ever made.

        Returns:
            list[Track]: Those tracks (track ids are their position in
                :attr:`tracks`).
        """
        return [self.tracks[track_id] for track_id in self.crops]

    @property
    def stopped(self) -> bool:
        """Whether the faces service has failed too often in a row to keep trying.

        Returns:
            bool: ``True`` after :data:`FAILED_BATCHES_TO_STOP` failed batches
                in a row.
        """
        return self.failed_in_a_row >= FAILED_BATCHES_TO_STOP

    @staticmethod
    def _is_good(detection: Detection, frontal: float) -> bool:
        """Whether a face is good enough to found a person on.

        Args:
            detection: The detection.
            frontal: Its frontality.

        Returns:
            bool: Confident, large enough and roughly frontal.
        """
        return (
            detection.score >= GOOD_SCORE
            and min(detection.box[2], detection.box[3]) >= GOOD_MIN_PX
            and frontal >= GOOD_FRONTALITY
        )

    def interval(self) -> float:
        """The interval the samples actually came at.

        Returns:
            float: The requested interval, or the median gap between samples
                when the sampler stretched it over a long clip.
        """
        gaps = [later - earlier for earlier, later in itertools.pairwise(self.sample_times)]
        positive = [gap for gap in gaps if gap > 0]
        return max(self.interval_s, statistics.median(positive)) if positive else self.interval_s

    def _rank(self, track: Track) -> float:
        """How prominent a track is: its duration times its mean area.

        The duration counts one sampling interval for the last sample; the
        interval is the effective one, fixed the first time any track is
        ranked, so every ranking of the video agrees.

        Args:
            track: The track.

        Returns:
            float: Higher is more prominent.
        """
        if self._ranking_interval is None:
            self._ranking_interval = self.interval()
        return (track.last_time - track.first_time + self._ranking_interval) * track.mean_area

    def _prune_closed(self) -> None:
        """Drop the crops of closed tracks that can no longer make the track cap.

        A closed track never grows, so once more than :data:`MAX_TRACKS`
        closed tracks hold crops, those ranked below them can never be among
        the final ones; dropping them bounds memory for a crowded video.
        """
        closed = [track for track in self.tracks_with_crops() if track.closed]
        if len(closed) <= MAX_TRACKS:
            return
        closed.sort(key=self._rank, reverse=True)
        for track in closed[MAX_TRACKS:]:
            del self.crops[track.id]
        self.truncated = True

    def finish(self) -> FacesResult:
        """Cap, group, merge and number the tracks into the result.

        Returns:
            FacesResult: The faces of the video.
        """
        if self.batches_answered == 0:
            return FacesResult(
                outcome="unavailable",
                grouped=False,
                truncated=False,
                entries=(),
                rows=(),
                frames_sent=self.frames_sent,
                tracks_sent=0,
            )
        interval = self.interval()
        kept = sorted(self.tracks_with_crops(), key=lambda track: track.id)
        if len(kept) > MAX_TRACKS:
            kept.sort(key=self._rank, reverse=True)
            kept = sorted(kept[:MAX_TRACKS], key=lambda track: track.id)
            self.truncated = True
        grouped, tracks_sent, labels = self._group(kept)
        members: dict[int, list[Track]] = {}
        drafts: list[tuple[bool, list[Track]]] = []
        for track in kept:
            label = labels.get(track.id)
            if label is None:
                drafts.append((False, [track]))
            else:
                members.setdefault(label, []).append(track)
        drafts.extend((True, tracks) for tracks in members.values())
        entries, rows = self._entries(drafts, interval)
        outcome: FacesOutcome = "partial" if self.batches_failed or self.undecodable else "ok"
        return FacesResult(
            outcome=outcome,
            grouped=grouped,
            truncated=self.truncated,
            entries=entries,
            rows=rows,
            frames_sent=self.frames_sent,
            tracks_sent=tracks_sent,
        )

    def _group(self, kept: Sequence[Track]) -> tuple[bool, int, dict[int, int]]:
        """Ask the service which tracks show the same person (``group`` mode).

        Each track goes with its best few crops: a track with a good face is
        an anchor and sends only its good ones; one without is weak and sends
        its best ones, and the service lets it join at most a person the
        anchors make up. When that request would be over the service's byte
        budget, each track goes with its single best crop; when even that is
        over, nothing is sent and every track stays its own entry.

        Args:
            kept: The tracks kept.

        Returns:
            tuple[bool, int, dict[int, int]]: Whether a grouping reply came
                back, how many tracks were sent, and the label of each track
                the service assigned to a person.
        """
        if self.mode != "group":
            return False, 0, {}
        eligible = [track for track in kept if self.crops[track.id].group]
        if not eligible:
            return False, 0, {}
        request: list[GroupTrack] | None = None
        for faces_per_track in (GROUP_FACES_PER_TRACK, 1):
            candidate: list[GroupTrack] = []
            for track in eligible:
                anchor, crops = self.crops[track.id].group_faces()
                candidate.append(
                    GroupTrack(
                        samples=tuple(detection.sample for detection in track.detections),
                        faces=tuple(crops[:faces_per_track]),
                        anchor=anchor,
                    )
                )
            if len(faces_api.group_request_body(candidate)) <= faces_api.MAX_REQUEST_BYTES:
                request = candidate
                break
        if request is None:
            logger.warning(
                "Face grouping skipped: {} tracks are over the request budget even with one face each.", len(eligible)
            )
            return False, 0, {}
        labels = group_faces(request, config=self.config)
        if labels is None:
            return False, len(eligible), {}
        return (
            True,
            len(eligible),
            {track.id: label for track, label in zip(eligible, labels, strict=True) if label >= 0},
        )

    def _clamped(self, spans: Sequence[tuple[float, float]], *, last_seen: float) -> list[tuple[float, float]]:
        """End each span at the clip's end at the latest, never before the face was last seen.

        Args:
            spans: Merged appearance spans.
            last_seen: Time of the entry's last detection.

        Returns:
            list[tuple[float, float]]: The spans; unchanged when the duration is unknown.
        """
        if self.duration_sec <= 0:
            return list(spans)
        return [(start, max(min(end, self.duration_sec), min(end, last_seen))) for start, end in spans]

    def _entries(
        self, drafts: Sequence[tuple[bool, list[Track]]], interval: float
    ) -> tuple[tuple[FaceEntry, ...], tuple[DetectionRow, ...]]:
        """Merge, cap and number the entries, and list their detections.

        Args:
            drafts: Per entry, whether it is a grouped person and its tracks.
            interval: The sampling interval.

        Returns:
            tuple[tuple[FaceEntry, ...], tuple[DetectionRow, ...]]: The
                entries by number, and their detections by time.
        """
        planned: list[tuple[float, float, float, bool, list[Track], list[tuple[float, float]]]] = []
        for assigned, tracks in drafts:
            spans = self._clamped(
                merge_spans([(t.first_time, t.last_time + interval) for t in tracks], gap=2 * interval),
                last_seen=max(t.last_time for t in tracks),
            )
            detections = [detection for track in tracks for detection in track.detections]
            area = sum(d.box[2] * d.box[3] for d in detections) / len(detections)
            first = min(tracks, key=lambda track: (track.first_time, track.detections[0].box[0]))
            prominence = sum(end - start for start, end in spans) * area
            planned.append((first.first_time, float(first.detections[0].box[0]), prominence, assigned, tracks, spans))
        if len(planned) > MAX_ENTRIES:
            planned.sort(key=lambda item: item[2], reverse=True)
            del planned[MAX_ENTRIES:]
            self.truncated = True
        planned.sort(key=lambda item: (item[0], item[1]))
        entries: list[FaceEntry] = []
        rows: list[DetectionRow] = []
        for number, (_first, _x, _prominence, assigned, tracks, spans) in enumerate(planned, start=1):
            bests = [self.crops[track.id] for track in tracks]
            top = max(bests, key=lambda kept: kept.best.quality if kept.best else -1.0)
            best = top.best
            if best is None:
                continue
            smalls = sorted((crop for kept in bests for crop in kept.small), key=lambda c: c.quality, reverse=True)
            crops = [c for c in smalls if (c.time_sec, c.box) != (best.time_sec, best.box)][:EXTRA_CROPS]
            entries.append(
                FaceEntry(
                    number=number,
                    assigned=assigned,
                    appearances=tuple(spans),
                    detections=sum(len(track.detections) for track in tracks),
                    best=best,
                    crops=tuple(crops),
                    thumbnail=top.thumbnail,
                )
            )
            rows.extend(
                DetectionRow(
                    time_sec=d.time_sec,
                    entry=number,
                    box=_to_source(d.box, self.scales.get(d.sample, 1.0)),
                    score=d.score,
                )
                for track in tracks
                for d in track.detections
            )
        rows.sort(key=lambda row: (row.time_sec, row.entry))
        return tuple(entries), tuple(rows)


def extract_faces(
    file_path: Path, *, mode: Literal["detect", "group"], interval_s: float, max_frames: int
) -> FacesResult | None:
    """Find the faces of a video, track them, and group them per person in ``group`` mode.

    Args:
        file_path: The uploaded media.
        mode: ``detect`` (boxes and crops only) or ``group`` (also grouped per
            person by the faces service).
        interval_s: Seconds between sampled frames.
        max_frames: Most frames to sample.

    Returns:
        FacesResult | None: The faces, or ``None`` when no frame could be
            sampled (an audio-only or undecodable file).
    """
    extraction = _Extraction(mode, interval_s, load_faces_env())
    batch: list[SampledFrame] = []
    batch_bytes = 0
    with contextlib.closing(sample_frames(file_path, interval_s=interval_s, max_frames=max_frames)) as frames:
        for frame in frames:
            size = _source_bytes(frame)
            if batch and batch_bytes + size > SOURCE_BATCH_BYTES:
                extraction.process(batch)
                batch = []
                batch_bytes = 0
                if extraction.stopped:
                    break
            batch.append(frame)
            batch_bytes += size
            if len(batch) >= BATCH_IMAGES:
                extraction.process(batch)
                batch = []
                batch_bytes = 0
                if extraction.stopped:
                    break
    if batch:
        extraction.process(batch)
    if not extraction.sample_times:
        return None
    result = extraction.finish()
    logger.info("Face stage judged {} frames of '{}' (outcome={}).", result.frames_sent, file_path.name, result.outcome)
    return result
