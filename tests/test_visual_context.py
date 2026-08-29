"""Tests for the visual-context agent (keyframe captioning)."""

import io
from collections.abc import Sequence
from typing import Any, override

import httpx2
import openai
import pytest
from PIL import Image

from nextext.core.keyframes import Keyframe
from nextext.core.openai_cfg import InferencePipeline
from nextext.core.visual_context import (
    FrameCaption,
    describe_keyframes,
    format_visual_context,
    prepare_frame,
)


def _jpeg(width: int, height: int, *, mode: str = "RGB") -> bytes:
    """Encode a solid test image.

    Args:
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        mode (str): Pillow mode to build the source image in.

    Returns:
        bytes: JPEG-encoded (or PNG for non-JPEG-able modes) image payload.
    """
    image = Image.new(mode, (width, height), "red" if mode != "L" else 128)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG" if mode in {"RGB", "L"} else "PNG")
    return buffer.getvalue()


def _size(payload: bytes) -> tuple[int, int]:
    """Return the pixel dimensions of an encoded image.

    Args:
        payload (bytes): Encoded image bytes.

    Returns:
        tuple[int, int]: ``(width, height)``.
    """
    with Image.open(io.BytesIO(payload)) as image:
        return image.size


class _FakePipeline(InferencePipeline):
    """InferencePipeline double returning queued replies from ``call_vision``."""

    def __init__(self, replies: Sequence[str | Exception]) -> None:
        """Store the queued replies.

        Args:
            replies (Sequence[str | Exception]): One entry per ``call_vision``
                call; an ``Exception`` is raised instead of returned. An
                exhausted queue returns ``""``.
        """
        self.replies = list(replies)
        self.calls: list[tuple[str, list[bytes]]] = []

    @override
    def load_prompt(self, keyword: str = "system") -> str:
        """Return a stub caption instruction.

        Args:
            keyword (str): Prompt keyword; expected to be ``"frame_caption"``.

        Returns:
            str: A fixed instruction string.
        """
        assert keyword == "frame_caption"
        return "describe the frame"

    @override
    def call_vision(
        self,
        prompt: str,
        images: Sequence[bytes],
        mime_type: str = "image/jpeg",
        model: str | None = None,
        temperature: float = 0.1,
        seed: int = 42,
        num_predict: int | None = None,
        system_prompt: str | None = None,
        include_system_prompt: bool = True,
        think: bool | None = None,
    ) -> str:
        """Record the call and return (or raise) the next queued reply.

        Args:
            prompt (str): Instruction text.
            images (Sequence[bytes]): Image payloads.
            mime_type (str): MIME type of the payloads.
            model (str | None): Unused.
            temperature (float): Unused.
            seed (int): Unused.
            num_predict (int | None): Unused.
            system_prompt (str | None): Unused.
            include_system_prompt (bool): Unused.
            think (bool | None): Unused.

        Returns:
            str: The queued reply.

        Raises:
            Exception: When the queued entry is an exception instance.
        """
        self.calls.append((prompt, list(images)))
        if not self.replies:
            return ""
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _status_error(code: int) -> openai.APIStatusError:
    """Build an ``APIStatusError`` carrying a given HTTP status.

    Args:
        code (int): HTTP status code to attach.

    Returns:
        openai.APIStatusError: An error the agent can classify.
    """
    request = httpx2.Request("POST", "http://inference.invalid/v1/chat/completions")
    response = httpx2.Response(code, request=request)
    return openai.APIStatusError("boom", response=response, body=None)


def _frames(count: int, *, spacing: float = 10.0) -> list[Keyframe]:
    """Build evenly spaced keyframe samples.

    Args:
        count (int): Number of samples.
        spacing (float): Seconds between samples.

    Returns:
        list[Keyframe]: Timestamped synthetic frames.
    """
    return [Keyframe(time_sec=i * spacing, jpeg=_jpeg(32, 32)) for i in range(count)]


# ---------------------------------------------------------------------------
# prepare_frame
# ---------------------------------------------------------------------------


def test_prepare_frame_downscales_oversized_images() -> None:
    """A frame larger than the budget is scaled so its longest edge fits."""
    prepared = prepare_frame(_jpeg(1920, 1080), max_side=512)
    assert max(_size(prepared)) == 512


def test_prepare_frame_preserves_aspect_ratio() -> None:
    """Downscaling must not distort the frame."""
    width, height = _size(prepare_frame(_jpeg(1600, 900), max_side=800))
    assert (width, height) == (800, 450)


def test_prepare_frame_leaves_small_images_at_their_size() -> None:
    """A frame already within budget is not upscaled."""
    assert _size(prepare_frame(_jpeg(320, 240), max_side=1024)) == (320, 240)


def test_prepare_frame_returns_jpeg_bytes() -> None:
    """Output is always JPEG, whatever the input encoding was."""
    assert prepare_frame(_jpeg(64, 64, mode="RGBA"), max_side=256).startswith(b"\xff\xd8")


def test_prepare_frame_failsoft_returns_input_on_undecodable_bytes() -> None:
    """Un-openable bytes are passed through rather than raising."""
    junk = b"not an image at all"
    assert prepare_frame(junk, max_side=256) == junk


# ---------------------------------------------------------------------------
# describe_keyframes
# ---------------------------------------------------------------------------


def test_describe_keyframes_captions_every_frame_in_order() -> None:
    """One caption per frame, carrying that frame's timestamp, in time order."""
    pipeline = _FakePipeline(["a hallway", "a whiteboard", "a street"])

    captions = describe_keyframes(_frames(3), pipeline, max_frames=10, max_side=512)

    assert [c.caption for c in captions] == ["a hallway", "a whiteboard", "a street"]
    assert [c.time_sec for c in captions] == [0.0, 10.0, 20.0]


def test_describe_keyframes_sends_one_image_per_request() -> None:
    """Each request carries exactly the frame being described."""
    pipeline = _FakePipeline(["one", "two"])

    describe_keyframes(_frames(2), pipeline, max_frames=10, max_side=512)

    assert len(pipeline.calls) == 2
    assert all(len(images) == 1 for _, images in pipeline.calls)


def test_describe_keyframes_respects_the_frame_budget() -> None:
    """More frames than the budget are subsampled, not all captioned."""
    pipeline = _FakePipeline(["x"] * 10)

    captions = describe_keyframes(_frames(10), pipeline, max_frames=3, max_side=512)

    assert len(pipeline.calls) == 3
    assert len(captions) == 3


def test_describe_keyframes_skips_a_frame_that_fails_transiently() -> None:
    """A per-frame outage costs that caption only; the rest still land."""
    pipeline = _FakePipeline(["first", openai.APIConnectionError(request=httpx2.Request("POST", "http://x")), "third"])

    captions = describe_keyframes(_frames(3), pipeline, max_frames=10, max_side=512)

    assert [c.caption for c in captions] == ["first", "third"]
    assert len(pipeline.calls) == 3


def test_describe_keyframes_aborts_when_the_model_rejects_the_first_image() -> None:
    """A 4xx on frame one means the model is not vision-capable — stop at once.

    Paying N failing round-trips per job against a text-only model would slow
    every summary down for nothing, so the loop must give up immediately.
    """
    pipeline = _FakePipeline([_status_error(400), "never reached"])

    captions = describe_keyframes(_frames(4), pipeline, max_frames=10, max_side=512)

    assert captions == []
    assert len(pipeline.calls) == 1


def test_describe_keyframes_continues_after_a_late_server_error() -> None:
    """A 5xx mid-run is transient — later frames are still attempted."""
    pipeline = _FakePipeline(["first", _status_error(503), "third"])

    captions = describe_keyframes(_frames(3), pipeline, max_frames=10, max_side=512)

    assert [c.caption for c in captions] == ["first", "third"]


def test_describe_keyframes_drops_empty_replies() -> None:
    """Blank captions carry no information and are omitted."""
    pipeline = _FakePipeline(["  ", "a room"])

    captions = describe_keyframes(_frames(2), pipeline, max_frames=10, max_side=512)

    assert [c.caption for c in captions] == ["a room"]


def test_describe_keyframes_drops_refusals() -> None:
    """A model that says it cannot see the image contributes no caption."""
    pipeline = _FakePipeline(["I'm unable to see the image you provided.", "a parked car"])

    captions = describe_keyframes(_frames(2), pipeline, max_frames=10, max_side=512)

    assert [c.caption for c in captions] == ["a parked car"]


def test_describe_keyframes_returns_empty_for_no_frames() -> None:
    """No frames means no requests and no captions."""
    pipeline = _FakePipeline([])

    assert describe_keyframes([], pipeline, max_frames=10, max_side=512) == []
    assert pipeline.calls == []


def test_describe_keyframes_never_raises_on_unexpected_errors() -> None:
    """Captioning is an enhancement: an unexpected failure must not fail the job."""
    pipeline = _FakePipeline([RuntimeError("provider unreachable"), "still fine"])

    captions = describe_keyframes(_frames(2), pipeline, max_frames=10, max_side=512)

    assert [c.caption for c in captions] == ["still fine"]


def test_describe_keyframes_uses_the_localized_caption_prompt() -> None:
    """The instruction comes from the ``frame_caption`` prompt file."""
    pipeline = _FakePipeline(["ok"])

    describe_keyframes(_frames(1), pipeline, max_frames=10, max_side=512)

    assert pipeline.calls[0][0] == "describe the frame"


# ---------------------------------------------------------------------------
# format_visual_context
# ---------------------------------------------------------------------------


def test_format_visual_context_labels_each_caption_with_mm_ss() -> None:
    """Captions render one per line, stamped with their moment in the clip."""
    block = format_visual_context(
        [FrameCaption(time_sec=0.0, caption="a hallway"), FrameCaption(time_sec=75.0, caption="a whiteboard")]
    )
    assert block == "[00:00] a hallway\n[01:15] a whiteboard"


def test_format_visual_context_handles_clips_over_an_hour() -> None:
    """Past 60 minutes the stamp grows rather than wrapping around."""
    assert format_visual_context([FrameCaption(time_sec=3725.0, caption="late")]) == "[62:05] late"


def test_format_visual_context_empty_is_empty_string() -> None:
    """No captions yields no block, so callers can test it for truthiness."""
    assert format_visual_context([]) == ""


def test_format_visual_context_collapses_multiline_captions() -> None:
    """A chatty model must not break the one-caption-per-line contract."""
    block = format_visual_context([FrameCaption(time_sec=5.0, caption="line one\nline two")])
    assert block == "[00:05] line one line two"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -3.0])
def test_format_visual_context_tolerates_odd_timestamps(value: float) -> None:
    """Non-finite or negative times still render a line rather than crashing.

    Args:
        value (float): A timestamp a broken container could produce.
    """
    block = format_visual_context([FrameCaption(time_sec=value, caption="frame")])
    assert block.endswith("frame")


def test_frame_caption_is_hashable_and_frozen() -> None:
    """Captions are value objects the worker can safely stash on job state."""
    caption = FrameCaption(time_sec=1.0, caption="x")
    with pytest.raises(Exception):  # noqa: B017 - dataclasses.FrozenInstanceError
        caption.caption = "y"  # type: ignore[misc]


def test_describe_keyframes_accepts_a_real_pipeline_type() -> None:
    """The agent's contract is the InferencePipeline surface, not a duck type."""
    assert isinstance(_FakePipeline([]), InferencePipeline)


def test_prepare_frame_is_applied_before_upload() -> None:
    """Oversized frames are downscaled on the way to the model, not sent raw."""
    big = Keyframe(time_sec=0.0, jpeg=_jpeg(2000, 2000))
    pipeline = _FakePipeline(["ok"])

    describe_keyframes([big], pipeline, max_frames=10, max_side=256)

    sent: Any = pipeline.calls[0][1][0]
    assert max(_size(sent)) == 256
