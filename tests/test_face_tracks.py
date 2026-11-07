"""Tests for face tracks, shot cuts and crop quality (pure geometry, no model)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from nextext.core.face_tracks import (
    Box,
    Detection,
    Track,
    assign_tracks,
    frontality,
    grey_thumb,
    iou,
    is_cut,
    merge_spans,
    quality,
    sharpness,
)

# Five landmarks of a frontal face in a 100 px box at the origin: eyes level,
# nose centred between them.
_FRONTAL = ((30.0, 40.0), (70.0, 40.0), (50.0, 60.0), (35.0, 75.0), (65.0, 75.0))
# The same face turned to one side: the nose sits over one eye.
_PROFILE = ((30.0, 40.0), (50.0, 40.0), (52.0, 60.0), (35.0, 75.0), (50.0, 75.0))


def _det(sample: int, box: Box, score: float = 0.9) -> Detection:
    """Build a detection at one second per sample.

    Args:
        sample: The sample index.
        box: The box in source pixels.
        score: The detector's confidence.

    Returns:
        Detection: The detection.
    """
    return Detection(sample=sample, time_sec=float(sample), box=box, score=score, landmarks=_FRONTAL)


def _run(samples: list[list[Detection]], cuts: frozenset[int] = frozenset()) -> tuple[list[Track], list[list[int]]]:
    """Feed samples of detections through the tracker.

    Args:
        samples: Detections per sample, in sample order (sample ``i`` at index ``i``).
        cuts: Samples that open a new shot.

    Returns:
        tuple[list[Track], list[list[int]]]: Every track, and the track id each
            detection was assigned to, per sample.
    """
    tracks: list[Track] = []
    assigned: list[list[int]] = []
    for index, detections in enumerate(samples):
        owners = assign_tracks(tracks, detections, sample=index, cut=index in cuts)
        assigned.append([track.id for track in owners])
    return tracks, assigned


def test_iou_identical_and_disjoint() -> None:
    """Identical boxes overlap fully, disjoint ones not at all, half-shifted ones by a third."""
    assert iou((10, 10, 100, 100), (10, 10, 100, 100)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (50, 50, 10, 10)) == 0.0
    assert iou((0, 0, 100, 100), (50, 0, 100, 100)) == pytest.approx(1 / 3)
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_steady_face_forms_one_track() -> None:
    """A face that drifts a little from sample to sample stays one track."""
    tracks, assigned = _run([[_det(i, (100 + 4 * i, 100, 80, 80))] for i in range(6)])

    assert len(tracks) == 1
    assert assigned == [[0]] * 6
    track = tracks[0]
    assert (track.first_time, track.last_time, track.last_sample) == (0.0, 5.0, 5)
    assert track.mean_area == pytest.approx(6400.0)


def test_gap_of_one_sample_keeps_track_two_breaks_it() -> None:
    """One sample without the face keeps its track; two missing samples start a new one."""
    box: Box = (100, 100, 80, 80)
    tracks, assigned = _run([[_det(0, box)], [], [_det(2, box)], [], [], [_det(5, box)]])

    assert assigned == [[0], [], [0], [], [], [1]]
    assert len(tracks) == 2
    assert tracks[0].closed


def test_cut_breaks_every_track() -> None:
    """At a shot cut every open track closes, even for a face in the same place."""
    box_a: Box = (100, 100, 80, 80)
    box_b: Box = (400, 100, 80, 80)
    tracks, assigned = _run(
        [[_det(0, box_a), _det(0, box_b)], [_det(1, box_a), _det(1, box_b)], [_det(2, box_a), _det(2, box_b)]],
        cuts=frozenset({2}),
    )

    assert assigned == [[0, 1], [0, 1], [2, 3]]
    assert tracks[0].closed and tracks[1].closed
    assert not tracks[2].closed and not tracks[3].closed


def test_size_jump_starts_new_track() -> None:
    """A box that grows past the size ratio is a new face, even though it overlaps the old one."""
    tracks, assigned = _run([[_det(0, (100, 100, 100, 100))], [_det(1, (100, 100, 130, 130))]])

    assert iou((100, 100, 100, 100), (100, 100, 130, 130)) >= 0.5
    assert assigned == [[0], [1]]
    assert len(tracks) == 2


def test_small_size_change_keeps_track() -> None:
    """A box that grows a little (a slow zoom) stays on its track."""
    _tracks, assigned = _run([[_det(0, (100, 100, 100, 100))], [_det(1, (100, 100, 110, 110))]])

    assert assigned == [[0], [0]]


def test_two_faces_two_tracks_no_swap() -> None:
    """Two faces keep their own tracks whatever order the detector lists them in."""
    left: list[Box] = [(100 + 5 * i, 100, 80, 80) for i in range(4)]
    right: list[Box] = [(220 - 5 * i, 100, 80, 80) for i in range(4)]
    samples = [
        [_det(0, left[0]), _det(0, right[0])],
        [_det(1, right[1]), _det(1, left[1])],
        [_det(2, left[2]), _det(2, right[2])],
        [_det(3, right[3]), _det(3, left[3])],
    ]

    tracks, assigned = _run(samples)

    assert len(tracks) == 2
    assert assigned == [[0, 1], [1, 0], [0, 1], [1, 0]]
    assert [d.box for d in tracks[0].detections] == left
    assert [d.box for d in tracks[1].detections] == right


def test_a_track_takes_one_detection_per_sample() -> None:
    """Two detections over one old face: one continues the track, the other opens a new one."""
    _tracks, assigned = _run(
        [[_det(0, (100, 100, 100, 100))], [_det(1, (100, 100, 100, 100)), _det(1, (105, 100, 100, 100))]]
    )

    assert sorted(assigned[1]) == [0, 1]


def test_frontality_frontal_vs_profile() -> None:
    """A nose centred between the eyes is frontal; one over an eye is not."""
    assert frontality(_FRONTAL) == pytest.approx(1.0)
    assert frontality(_PROFILE) < 0.5
    assert frontality(((10.0, 10.0), (10.0, 10.0), (10.0, 20.0))) == 0.0
    assert frontality(()) == 0.0


def test_sharpness_flat_vs_checkerboard() -> None:
    """A flat patch has no detail; a checkerboard is as sharp as it gets."""
    flat = np.full((32, 32), 128.0, dtype=np.float32)
    checker = (np.indices((32, 32)).sum(axis=0) % 2 * 255).astype(np.float32)

    assert sharpness(flat) == 0.0
    assert sharpness(checker) == pytest.approx(1.0)
    assert sharpness(np.zeros((2, 2), dtype=np.float32)) == 0.0


def test_quality_orders_larger_sharper_frontal_first() -> None:
    """Quality rises with size (up to 112 px), frontality and sharpness, and scales with score."""
    best = quality(0.95, 120, 1.0, 1.0)
    assert best == pytest.approx(0.95)
    assert quality(0.95, 56, 1.0, 1.0) < best
    assert quality(0.95, 120, 0.0, 1.0) < best
    assert quality(0.95, 120, 1.0, 0.0) < best
    assert quality(0.95, 56, 0.0, 0.0) == pytest.approx(0.95 * 0.5 * 0.5 * 0.5)


def test_merge_spans_joins_within_gap() -> None:
    """Spans closer than the gap merge; farther ones stay apart; input order does not matter."""
    spans = [(10.0, 12.0), (0.0, 3.0), (4.0, 6.0), (30.0, 31.0)]

    assert merge_spans(spans, gap=2.0) == [(0.0, 6.0), (10.0, 12.0), (30.0, 31.0)]
    assert merge_spans([(0.0, 5.0), (1.0, 2.0)], gap=0.0) == [(0.0, 5.0)]
    assert merge_spans([], gap=2.0) == []


def test_grey_thumb_is_small_float_grid() -> None:
    """The cut detector compares 32x18 grey thumbnails."""
    thumb = grey_thumb(Image.new("RGB", (640, 360), (200, 200, 200)))

    assert thumb.shape == (18, 32)
    assert thumb.dtype == np.float32
    assert float(thumb.mean()) == pytest.approx(200.0, abs=1.0)


def test_is_cut_on_scene_change_not_on_noise() -> None:
    """A new scene is a cut; sensor noise on the same scene is not; the first sample never is."""
    rng = np.random.default_rng(7)
    scene = np.zeros((360, 640, 3), dtype=np.uint8)
    scene[:, :320] = 40
    scene[:, 320:] = 180
    noisy = np.clip(scene.astype(np.int16) + rng.integers(-8, 9, scene.shape), 0, 255).astype(np.uint8)
    other = np.full((360, 640, 3), 230, dtype=np.uint8)
    other[:, 320:] = 20

    first = grey_thumb(Image.fromarray(scene))
    assert not is_cut(None, first)
    assert not is_cut(first, grey_thumb(Image.fromarray(noisy)))
    assert is_cut(first, grey_thumb(Image.fromarray(other)))
