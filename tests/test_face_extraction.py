"""Tests for face extraction: batching, filtering, tracking, grouping and crops.

The frame sampler and the faces client are replaced by fakes: frames are flat
synthetic Pillow images, and "faces" are boxes the fake detector reports at
planned positions. No real face, photo or video is involved.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import httpx
import pytest
from PIL import Image

from nextext import pipeline
from nextext.core import face_extraction, faces
from nextext.core.face_extraction import FacesResult, extract_faces
from nextext.core.face_tracks import Box
from nextext.core.faces import DetectedFace, GroupTrack
from nextext.core.keyframes import SampledFrame

W, H = 640, 360
DARK, LIGHT = 60, 200


def _face(box: Box, score: float = 0.95, size: tuple[int, int] = (W, H)) -> DetectedFace:
    """Describe a frontal face at a box, normalized to the frame as the service reports it.

    Args:
        box: ``(x, y, w, h)`` in source pixels.
        score: The detector's confidence.
        size: The frame's size.

    Returns:
        DetectedFace: The face.
    """
    x, y, w, h = box
    sw, sh = size
    points = ((0.3, 0.4), (0.7, 0.4), (0.5, 0.6), (0.35, 0.8), (0.65, 0.8))
    return DetectedFace(
        box=(x / sw, y / sh, w / sw, h / sh),
        score=score,
        landmarks=tuple(((x + px * w) / sw, (y + py * h) / sh) for px, py in points),
    )


class _Fakes:
    """Stand-ins for the sampler and the faces service, with what they were sent."""

    def __init__(
        self,
        scenes: Sequence[int],
        planned: Sequence[list[DetectedFace] | None],
        *,
        size: tuple[int, int] = (W, H),
        scale: float = 1.0,
        step: float = 1.0,
        fail_calls: frozenset[int] = frozenset(),
        labels: Callable[[list[GroupTrack]], list[int] | None] | None = None,
    ) -> None:
        """Plan a clip.

        Args:
            scenes: Grey level of each sample's frame; a jump is a shot cut.
            planned: Faces the detector reports per sample (``None``: undecodable).
            size: Frame size.
            scale: How far the sampler scaled each frame down from the source.
            step: Seconds between samples (the sampler's effective interval).
            fail_calls: Detect calls (by number) that fail.
            labels: Replies to ``/faces/group``; ``None`` makes it fail.
        """
        self.scenes = list(scenes)
        self.planned = list(planned)
        self.size = size
        self.scale = scale
        self.step = step
        self.fail_calls = fail_calls
        self.labels = labels
        self.batches: list[list[bytes]] = []
        self.group_calls: list[list[GroupTrack]] = []
        self.yielded = 0
        self._cursor = 0

    def sample_frames(
        self, file_path: Path, *, interval_s: float, max_frames: int, **kwargs: Any
    ) -> Iterator[SampledFrame]:
        """Yield the planned frames, one per second.

        Args:
            file_path: Ignored.
            interval_s: Ignored.
            max_frames: Ignored.
            **kwargs: Ignored.

        Yields:
            SampledFrame: The frames.
        """
        for index, level in enumerate(self.scenes):
            image = Image.new("RGB", self.size, (level,) * 3)
            self.yielded += 1
            yield SampledFrame(index=index, time_sec=index * self.step, image=image, scale=self.scale)

    def detect_faces(
        self, images: Sequence[bytes], *, min_score: float = 0.7, config: object = None
    ) -> list[list[DetectedFace] | None] | None:
        """Report the planned faces of the next samples.

        Args:
            images: The batch.
            min_score: Ignored.
            config: Ignored.

        Returns:
            list[list[DetectedFace] | None] | None: The planned faces, or ``None``
                for a failing call.
        """
        call = len(self.batches)
        self.batches.append(list(images))
        start = self._cursor
        self._cursor += len(images)
        if call in self.fail_calls:
            return None
        return [None if faces is None else list(faces) for faces in self.planned[start : start + len(images)]]

    def group_faces(self, tracks: Sequence[GroupTrack], *, config: object = None) -> list[int] | None:
        """Answer a grouping request.

        Args:
            tracks: The tracks sent.
            config: Ignored.

        Returns:
            list[int] | None: The planned labels.
        """
        self.group_calls.append(list(tracks))
        return None if self.labels is None else self.labels(list(tracks))

    def install(self, monkeypatch: pytest.MonkeyPatch) -> _Fakes:
        """Patch the extraction module to use these fakes.

        Args:
            monkeypatch: Pytest patcher.

        Returns:
            _Fakes: ``self``.
        """
        monkeypatch.setattr(face_extraction, "sample_frames", self.sample_frames)
        monkeypatch.setattr(face_extraction, "detect_faces", self.detect_faces)
        monkeypatch.setattr(face_extraction, "group_faces", self.group_faces)
        return self


def _run(mode: Literal["detect", "group"] = "group") -> FacesResult:
    """Run the extraction over the fake clip.

    Args:
        mode: ``detect`` or ``group``.

    Returns:
        FacesResult: The result, asserted present.
    """
    result = extract_faces(Path("synthetic.mp4"), mode=mode, interval_s=1.0, max_frames=600)
    assert result is not None
    return result


def _decode(jpeg: bytes) -> Image.Image:
    """Decode a JPEG, checking it is one.

    Args:
        jpeg: The bytes.

    Returns:
        Image.Image: The image.
    """
    image = Image.open(io.BytesIO(jpeg))
    assert image.format == "JPEG"
    return image


def _two_shots() -> tuple[list[int], list[list[DetectedFace] | None]]:
    """One face in samples 0-1 and again in 6-7, after a cut, with nothing between.

    Returns:
        tuple[list[int], list[list[DetectedFace] | None]]: Scenes and faces.
    """
    box: Box = (200, 100, 100, 100)
    scenes = [DARK] * 6 + [LIGHT] * 2
    planned: list[list[DetectedFace] | None] = [[_face(box)], [_face(box)], [], [], [], [], [_face(box)], [_face(box)]]
    return scenes, planned


def test_one_person_two_shots_groups_to_person_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two tracks the service labels alike become Person 1 with two appearances."""
    scenes, planned = _two_shots()
    fakes = _Fakes(scenes, planned, labels=lambda tracks: [0] * len(tracks)).install(monkeypatch)

    result = _run("group")

    assert (result.outcome, result.grouped, result.truncated) == ("ok", True, False)
    assert len(result.entries) == 1
    entry = result.entries[0]
    assert (entry.number, entry.assigned, entry.detections) == (1, True, 4)
    assert entry.appearances == ((0.0, 2.0), (6.0, 8.0))
    assert [(row.time_sec, row.entry) for row in result.rows] == [(0.0, 1), (1.0, 1), (6.0, 1), (7.0, 1)]
    assert (result.frames_sent, result.tracks_sent) == (8, 2)
    assert len(fakes.group_calls) == 1


def test_detect_mode_never_calls_group(monkeypatch: pytest.MonkeyPatch) -> None:
    """In detect mode nothing is grouped: each track is its own unassigned entry."""
    scenes, planned = _two_shots()

    def no_group(tracks: list[GroupTrack]) -> list[int]:
        raise AssertionError("detect mode must not group")

    _Fakes(scenes, planned, labels=no_group).install(monkeypatch)

    result = _run("detect")

    assert result.grouped is False
    assert [(e.number, e.assigned) for e in result.entries] == [(1, False), (2, False)]
    assert result.tracks_sent == 0


def test_group_failure_falls_back_to_tracks(monkeypatch: pytest.MonkeyPatch) -> None:
    """When grouping fails, every track stays a single, unassigned entry."""
    scenes, planned = _two_shots()
    _Fakes(scenes, planned, labels=None).install(monkeypatch)

    result = _run("group")

    assert result.grouped is False
    assert [(e.number, e.assigned, e.appearances) for e in result.entries] == [
        (1, False, ((0.0, 2.0),)),
        (2, False, ((6.0, 8.0),)),
    ]
    assert result.tracks_sent == 2


def test_unlabelled_track_stays_single(monkeypatch: pytest.MonkeyPatch) -> None:
    """A track the service labels -1 stays its own unassigned entry beside the grouped ones."""
    scenes, planned = _two_shots()
    _Fakes(scenes, planned, labels=lambda tracks: [0, -1]).install(monkeypatch)

    result = _run("group")

    assert result.grouped is True
    assert [(e.number, e.assigned) for e in result.entries] == [(1, True), (2, False)]


def test_cooccurring_tracks_send_their_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each track sent for grouping carries its sample indices and up to three good crops."""
    left: Box = (50, 100, 100, 100)
    right: Box = (400, 100, 100, 100)
    planned: list[list[DetectedFace] | None] = [[_face(left), _face(right)] for _ in range(5)]
    fakes = _Fakes([DARK] * 5, planned, labels=lambda tracks: [0, 1]).install(monkeypatch)

    result = _run("group")

    sent = fakes.group_calls[0]
    assert [track.samples for track in sent] == [(0, 1, 2, 3, 4), (0, 1, 2, 3, 4)]
    assert all(len(track.faces) == 3 for track in sent)
    for track in sent:
        for face in track.faces:
            image = _decode(face.jpeg)
            assert max(image.size) <= 160
            assert len(face.landmarks) == 5
            assert all(0 <= x <= image.width and 0 <= y <= image.height for x, y in face.landmarks)
    assert [e.assigned for e in result.entries] == [True, True]


def test_poor_faces_are_not_sent_for_grouping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tracks with no good face (low score or too small) stay single and are not sent."""
    planned: list[list[DetectedFace] | None] = [[_face((50, 100, 100, 100), score=0.75), _face((400, 100, 50, 50))]] * 3
    fakes = _Fakes([DARK] * 3, planned, labels=lambda tracks: [0] * len(tracks)).install(monkeypatch)

    result = _run("group")

    assert fakes.group_calls == []
    assert result.tracks_sent == 0
    assert result.grouped is False
    assert [e.assigned for e in result.entries] == [False, False]


def test_numbering_follows_first_appearance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Entries are numbered by first appearance, left to right on a tie."""
    late: Box = (50, 100, 80, 80)
    middle: Box = (250, 100, 80, 80)
    right: Box = (450, 100, 80, 80)
    planned: list[list[DetectedFace] | None] = [
        [_face(right), _face(middle)],
        [_face(right), _face(middle)],
        [],
        [_face(late)],
        [_face(late)],
    ]
    _Fakes([DARK] * 5, planned).install(monkeypatch)

    result = _run("detect")

    assert [(e.number, e.best.box[0], e.appearances[0][0]) for e in result.entries] == [
        (1, 250, 0.0),
        (2, 450, 0.0),
        (3, 50, 3.0),
    ]


def test_best_crop_is_highest_quality_with_margin_and_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The best shot is the highest-quality detection, cut with a wide margin and at most 1024 px."""
    size = (2000, 1200)
    box: Box = (900, 500, 200, 200)
    planned: list[list[DetectedFace] | None] = [
        [_face(box, score=0.80, size=size)],
        [_face(box, score=0.99, size=size)],
        [_face(box, score=0.85, size=size)],
    ]
    _Fakes([DARK] * 3, planned, size=size).install(monkeypatch)

    entry = _run("detect").entries[0]

    assert (entry.best.time_sec, entry.best.score, entry.best.box) == (1.0, 0.99, box)
    assert entry.best.quality >= max(crop.quality for crop in entry.crops)
    assert _decode(entry.best.jpeg).size == (400, 400)


def test_best_crop_of_a_large_face_is_downscaled(monkeypatch: pytest.MonkeyPatch) -> None:
    """A face whose margin crop exceeds 1024 px is scaled to 1024 on its long side."""
    size = (2000, 1200)
    planned: list[list[DetectedFace] | None] = [[_face((500, 100, 1000, 1000), size=size)]]
    _Fakes([DARK], planned, size=size).install(monkeypatch)

    best = _decode(_run("detect").entries[0].best.jpeg)

    assert max(best.size) == 1024
    assert best.size == (1024, 614)


def test_edge_face_crop_is_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Faces touching the frame edge give crops that stay inside the frame."""
    corner: Box = (540, 260, 100, 100)
    edge: Box = (0, 100, 100, 100)
    planned: list[list[DetectedFace] | None] = [[_face(edge), _face(corner)]]
    _Fakes([DARK], planned).install(monkeypatch)

    entries = _run("detect").entries

    by_x = {entry.best.box[0]: entry for entry in entries}
    assert _decode(by_x[0].best.jpeg).size == (150, 200)
    assert _decode(by_x[540].best.jpeg).size == (150, 150)
    for entry in entries:
        thumb = _decode(entry.thumbnail)
        assert thumb.width <= W and thumb.height <= H


def test_box_past_the_frame_is_clamped_to_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reported box that runs past the frame is cut back to the frame."""
    planned: list[list[DetectedFace] | None] = [[_face((600, 300, 100, 100))]]
    _Fakes([DARK], planned).install(monkeypatch)

    result = _run("detect")

    assert result.entries[0].best.box == (600, 300, 40, 60)


def test_thumbnail_and_crops_are_bounded_jpegs_without_exif(monkeypatch: pytest.MonkeyPatch) -> None:
    """Thumbnails stay within 128 px, crops within 256 px, three of them, all plain JPEGs."""
    size = (1280, 720)
    box: Box = (400, 100, 400, 400)
    planned: list[list[DetectedFace] | None] = [[_face(box, score=0.9 + i / 100, size=size)] for i in range(6)]
    _Fakes([DARK] * 6, planned, size=size).install(monkeypatch)

    entry = _run("detect").entries[0]

    assert len(entry.crops) == 3
    assert all(crop.time_sec != entry.best.time_sec for crop in entry.crops)
    assert [crop.quality for crop in entry.crops] == sorted((crop.quality for crop in entry.crops), reverse=True)
    images: list[tuple[bytes, int]] = [(entry.thumbnail, 128), (entry.best.jpeg, 1024)]
    images += [(crop.jpeg, 256) for crop in entry.crops]
    for jpeg, limit in images:
        image = _decode(jpeg)
        assert max(image.size) <= limit
        assert "exif" not in image.info
        assert len(image.getexif()) == 0
    assert max(_decode(entry.thumbnail).size) == 128


def test_crowd_is_capped_and_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Thirty faces a frame over a cut every sample stay bounded and mark the result truncated."""
    cells = [(20 + 105 * column, 5 + 72 * row) for row in range(5) for column in range(6)]
    planned: list[list[DetectedFace] | None] = [
        [_face((x, y, 64, 64), score=0.86 + index / 300) for index, (x, y) in enumerate(cells)] for _ in range(30)
    ]
    scenes = [DARK if index % 2 == 0 else LIGHT for index in range(30)]
    fakes = _Fakes(scenes, planned, labels=lambda tracks: [-1] * len(tracks)).install(monkeypatch)

    result = _run("group")

    assert result.truncated is True
    assert len(fakes.group_calls[0]) <= 500
    assert result.tracks_sent <= 500
    assert len(result.entries) <= 100
    per_time: dict[float, int] = {}
    for row in result.rows:
        per_time[row.time_sec] = per_time.get(row.time_sec, 0) + 1
    assert max(per_time.values()) <= 20
    assert [e.number for e in result.entries] == list(range(1, len(result.entries) + 1))


def test_too_many_faces_in_one_frame_keeps_the_most_prominent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the 20 faces with the highest score times area are kept from a crowded frame."""
    small = [_face((10 + 30 * i, 10, 40, 40), score=0.99) for i in range(20)]
    large = [_face((10 + 100 * i, 200, 90, 90), score=0.9) for i in range(5)]
    _Fakes([DARK], [small + large]).install(monkeypatch)

    result = _run("detect")

    assert result.truncated is True
    assert len(result.entries) == 20
    assert sum(1 for e in result.entries if e.best.box[2] == 90) == 5


def test_faces_below_40_px_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Faces smaller than 40 px are not kept, and that alone does not truncate."""
    _Fakes([DARK], [[_face((10, 10, 39, 60)), _face((200, 10, 60, 60))]]).install(monkeypatch)

    result = _run("detect")

    assert [e.best.box for e in result.entries] == [(200, 10, 60, 60)]
    assert result.truncated is False


def test_batches_of_eight_scaled_for_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Frames go out eight at a time, each scaled to at most 1280 px."""
    size = (1920, 1080)
    fakes = _Fakes([DARK] * 10, [[] for _ in range(10)], size=size).install(monkeypatch)

    result = _run("detect")

    assert [len(batch) for batch in fakes.batches] == [8, 2]
    assert all(_decode(jpeg).size == (1280, 720) for batch in fakes.batches for jpeg in batch)
    assert result.outcome == "ok"
    assert result.entries == ()


def test_batches_respect_the_source_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Large frames are batched so the source frames held at once stay within the budget."""
    size = (3840, 2160)
    fakes = _Fakes([DARK] * 6, [[] for _ in range(6)], size=size).install(monkeypatch)

    _run("detect")

    assert [len(batch) for batch in fakes.batches] == [4, 2]


def test_partial_outage_is_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    """When one batch fails the others still yield faces, and the result says partial."""
    box: Box = (200, 100, 100, 100)
    _Fakes([DARK] * 16, [[_face(box)] for _ in range(16)], fail_calls=frozenset({1})).install(monkeypatch)

    result = _run("detect")

    assert result.outcome == "partial"
    assert result.frames_sent == 16
    assert [e.appearances for e in result.entries] == [((0.0, 8.0),)]


def test_undecodable_image_is_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    """An image the service could not decode makes the result partial."""
    box: Box = (200, 100, 100, 100)
    _Fakes([DARK] * 3, [[_face(box)], None, [_face(box)]]).install(monkeypatch)

    result = _run("detect")

    assert result.outcome == "partial"
    assert len(result.entries) == 1


def test_total_outage_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no batch is answered the result is unavailable, with no entries."""
    box: Box = (200, 100, 100, 100)
    _Fakes([DARK] * 10, [[_face(box)] for _ in range(10)], fail_calls=frozenset({0, 1})).install(monkeypatch)

    result = _run("group")

    assert (result.outcome, result.grouped, result.entries, result.rows) == ("unavailable", False, (), ())
    assert result.frames_sent == 10


def test_no_frames_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A file without a single sampled frame has no face result at all."""
    fakes = _Fakes([], []).install(monkeypatch)

    assert extract_faces(Path("synthetic.wav"), mode="group", interval_s=1.0, max_frames=600) is None
    assert fakes.batches == []


def test_group_body_over_budget_is_ungrouped(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the grouping request would be over budget nothing is sent and every track stays single."""
    scenes, planned = _two_shots()
    _Fakes(scenes, planned).install(monkeypatch)
    monkeypatch.setattr(face_extraction, "group_faces", faces.group_faces)
    monkeypatch.setenv("FACES_API_BASE", "http://router.invalid")
    monkeypatch.setattr(faces, "MAX_REQUEST_BYTES", 64)

    def no_request(url: str, **kwargs: Any) -> None:
        raise AssertionError("an over-budget body must not be sent")

    monkeypatch.setattr(faces.httpx, "post", no_request)

    result = _run("group")

    assert result.grouped is False
    assert [e.assigned for e in result.entries] == [False, False]


def test_faces_pipeline_reads_the_sampling_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """faces_pipeline samples at FACES_INTERVAL_S / FACES_MAX_FRAMES and passes the mode on."""
    seen: dict[str, Any] = {}

    def fake_extract(file_path: Path, **kwargs: Any) -> None:
        seen.update(kwargs, file_path=file_path)

    monkeypatch.setattr(pipeline, "extract_faces", fake_extract)
    monkeypatch.setenv("FACES_INTERVAL_S", "2.5")
    monkeypatch.setenv("FACES_MAX_FRAMES", "40")

    assert pipeline.faces_pipeline(Path("synthetic.mp4"), mode="detect") is None
    assert seen == {"file_path": Path("synthetic.mp4"), "mode": "detect", "interval_s": 2.5, "max_frames": 40}


def test_boxes_are_reported_in_original_source_pixels(monkeypatch: pytest.MonkeyPatch) -> None:
    """A frame the sampler scaled down still reports boxes in the source's own pixels."""
    planned: list[list[DetectedFace] | None] = [[_face((100, 100, 80, 80))], [_face((100, 100, 80, 80))]]
    _Fakes([DARK] * 2, planned, scale=0.5).install(monkeypatch)

    result = _run("detect")

    entry = result.entries[0]
    assert entry.best.box == (200, 200, 160, 160)
    assert all(crop.box == (200, 200, 160, 160) for crop in entry.crops)
    assert [row.box for row in result.rows] == [(200, 200, 160, 160)] * 2
    assert _decode(entry.best.jpeg).size == (160, 160)


def test_faces_config_is_read_once_per_video(monkeypatch: pytest.MonkeyPatch, log_lines: list[str]) -> None:
    """The faces endpoint settings are read once per video, so a bad value warns once, not per batch."""
    fakes = _Fakes([DARK] * 16, [[] for _ in range(16)])
    monkeypatch.setattr(face_extraction, "sample_frames", fakes.sample_frames)
    monkeypatch.setenv("FACES_API_BASE", "http://router.invalid")
    monkeypatch.setenv("FACES_TIMEOUT", "soon")
    calls: list[float] = []

    def post(url: str, **kwargs: Any) -> httpx.Response:
        calls.append(kwargs["timeout"])
        images = kwargs["json"]["images"]
        return httpx.Response(
            200, json={"results": [{"faces": []} for _ in images]}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(faces.httpx, "post", post)

    result = _run("detect")

    assert result.outcome == "ok"
    assert len(calls) == 2
    assert sum("FACES_TIMEOUT" in line for line in log_lines) == 1


def test_three_failed_batches_in_a_row_stop_the_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    """After three failed batches in a row nothing more is sent, and nothing more is decoded."""
    box: Box = (200, 100, 100, 100)
    fakes = _Fakes([DARK] * 80, [[_face(box)] for _ in range(80)], fail_calls=frozenset(range(10))).install(monkeypatch)

    result = _run("detect")

    assert len(fakes.batches) == 3
    assert fakes.yielded == 24
    assert (result.outcome, result.entries, result.frames_sent) == ("unavailable", (), 24)


def test_breaker_after_a_success_is_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the service dies after answering, what it answered is kept and the result is partial."""
    box: Box = (200, 100, 100, 100)
    fakes = _Fakes([DARK] * 80, [[_face(box)] for _ in range(80)], fail_calls=frozenset(range(1, 10))).install(
        monkeypatch
    )

    result = _run("detect")

    assert len(fakes.batches) == 4
    assert result.outcome == "partial"
    assert [e.appearances for e in result.entries] == [((0.0, 8.0),)]


def test_an_answered_batch_resets_the_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Failures that never come three in a row do not stop the stage."""
    fakes = _Fakes([DARK] * 48, [[] for _ in range(48)], fail_calls=frozenset({0, 1, 3, 4})).install(monkeypatch)

    result = _run("detect")

    assert len(fakes.batches) == 6
    assert result.outcome == "partial"


def test_pruning_and_the_track_cap_rank_alike(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tracks dropped while reading and at the end are ranked by the same, effective interval.

    The sampler stretched its interval to 5 s. Ranked by duration plus 5 s
    times area, the large one-off face and the mid-size one beat the small
    long-lived one; ranked by the requested 1 s it would be the other way.
    """
    monkeypatch.setattr(face_extraction, "MAX_TRACKS", 2)
    large = _face((50, 100, 100, 100))
    small = _face((400, 100, 40, 40))
    middle = _face((250, 100, 80, 80))
    planned: list[list[DetectedFace] | None] = [[large, small], [small], [small, middle], [], [], [], [], []]
    _Fakes([DARK] * 8, planned, step=5.0).install(monkeypatch)

    result = _run("detect")

    assert result.truncated is True
    assert sorted(entry.best.box[0] for entry in result.entries) == [50, 250]
