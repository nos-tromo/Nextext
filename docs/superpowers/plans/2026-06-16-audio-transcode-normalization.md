# Audio Transcode Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-encode every upload to 16 kHz mono FLAC (via PyAV) before the external Whisper call, so libsndfile-only Whisper servers can always decode it.

**Architecture:** A new stateless agent `nextext/core/audio.py` exposes a `normalize_for_transcription` context manager that decodes any container/codec with PyAV and writes a temporary 16 kHz mono FLAC. `ExternalWhisperTranscriber.transcription()` wraps its upload with it (VAD still screens the original first). Both entry paths — the API job worker and `nextext-cli` — inherit the fix because both call `transcription_pipeline`. Normalization is fail-closed: an undecodable input raises `AudioDecodeError` instead of forwarding unusable bytes.

**Tech Stack:** Python 3.12, PyAV (`av`, bundled ffmpeg), pytest, loguru.

**Spec:** `docs/superpowers/specs/2026-06-16-audio-transcode-normalization-design.md`

---

## File Structure

- **Create** `nextext/core/audio.py` — the audio-normalization agent (`AudioDecodeError`, `normalize_for_transcription`, private `_transcode_to_flac`). One responsibility: turn any upload into a Whisper-decodable temp file.
- **Create** `tests/test_audio.py` — unit tests for the agent (round-trip + cleanup + failure).
- **Modify** `nextext/core/transcription.py` — import and wrap the Whisper upload with the normalizer; move the size log/cap onto the normalized file.
- **Modify** `tests/test_transcription.py` — stub the new seam in the two existing `transcription()` tests; add wiring + fail-closed tests.
- **Modify** `pyproject.toml` (+ regenerate `uv.lock`) — add `av>=12`.
- **Modify** `AGENTS.md`, `CLAUDE.md`, `docker/Dockerfile.backend` — update the now-stale "no local audio tooling" statements.

---

## Task 1: Add the PyAV dependency

**Files:**
- Modify: `pyproject.toml:13-42` (the `[project].dependencies` list)
- Modify: `uv.lock` (regenerated)

- [ ] **Step 1: Add `av>=12` to dependencies**

In `pyproject.toml`, insert the `av` entry into the alphabetically-sorted
`dependencies` list, right after `"arabic-reshaper>=3.0.0",`:

```toml
    "anyio>=4.4",
    "arabic-reshaper>=3.0.0",
    "av>=12",
    "camel-tools>=1.5.2",
```

- [ ] **Step 2: Regenerate the lockfile and install**

Run:
```bash
uv lock && uv sync --group dev
```
Expected: `uv.lock` updates with `av` (and its bundled-ffmpeg wheel); install succeeds.

- [ ] **Step 3: Verify the import works and pulls no torch**

Run:
```bash
uv run python -c "import av; print('av', av.__version__)"
uv run pytest tests/test_no_torch.py -q
```
Expected: prints an `av 12.x`/`13.x`/`14.x` version; `test_no_torch.py` PASSES (PyAV has no torch dependency, so the invariant holds).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add PyAV (av) for client-side audio normalization

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Create the audio-normalization agent

**Files:**
- Create: `nextext/core/audio.py`
- Test: `tests/test_audio.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_audio.py`. The source clip is synthesized with the stdlib
`wave` module (44.1 kHz **stereo**) rather than via PyAV: the normalizer is
codec-agnostic (it decodes everything through PyAV/ffmpeg), so a stereo 44.1 kHz
WAV exercises the real logic under test — resample 44.1 k→16 k, downmix
stereo→mono, and FLAC re-encode — with no encoder-availability or PyAV-typing
flakiness in the fixture. PyAV is used only to *inspect* the output.

```python
"""Tests for the audio-normalization agent (nextext.core.audio)."""

import wave
from pathlib import Path

import av
import numpy as np
import pytest

from nextext.core import audio


def _write_stereo_wav(path: Path, *, seconds: float = 0.4, rate: int = 44100) -> None:
    """Write a short 44.1 kHz stereo PCM WAV sine clip for decode tests.

    Stereo @ 44.1 kHz forces the normalizer's downmix-to-mono and
    rate-conversion paths to run.

    Args:
        path (Path): Output WAV path; its parent must exist.
        seconds (float): Clip duration in seconds.
        rate (int): Sample rate in Hz.
    """
    n = int(rate * seconds)
    t = np.linspace(0, seconds, n, endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype("<i2")
    stereo = np.repeat(tone[:, None], 2, axis=1)  # (n, 2) interleaved L == R
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(stereo.tobytes())


def test_normalize_produces_16k_mono_flac(tmp_path: Path) -> None:
    """A stereo 44.1 kHz source yields a 16 kHz mono FLAC, removed on exit.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    src = tmp_path / "sample.wav"
    _write_stereo_wav(src)

    captured: Path | None = None
    with audio.normalize_for_transcription(src) as flac_path:
        captured = flac_path
        assert flac_path.exists()
        assert flac_path.suffix == ".flac"
        with av.open(str(flac_path)) as container:
            stream = container.streams.audio[0]
            assert stream.codec_context.name == "flac"
            frame = next(container.decode(stream))
            assert frame.sample_rate == 16000
            assert frame.layout.name == "mono"

    assert captured is not None
    assert not captured.exists()  # temp file cleaned up on context exit


def test_normalize_raises_on_undecodable_input(tmp_path: Path) -> None:
    """Non-audio bytes raise AudioDecodeError (fail-closed).

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    bogus = tmp_path / "broken.ogg"
    bogus.write_bytes(b"NOT-REAL-AUDIO" * 64)
    with pytest.raises(audio.AudioDecodeError):
        with audio.normalize_for_transcription(bogus):
            pass
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
uv run pytest tests/test_audio.py -q
```
Expected: FAIL / collection error — `ModuleNotFoundError: No module named 'nextext.core.audio'` (or `AttributeError` on `audio.normalize_for_transcription`).

- [ ] **Step 3: Write the implementation**

Create `nextext/core/audio.py`:

```python
"""Audio-normalization agent: re-encode any upload to Whisper-ready audio.

The external Whisper endpoint (vLLM behind LiteLLM) decodes uploads with
``soundfile``/libsndfile only, with no ffmpeg fallback, so it rejects
containers/codecs libsndfile cannot read (Ogg Opus, m4a, mp4, mkv, webm, ...)
with an HTTP 400 "Invalid or unsupported audio file". This agent decodes the
upload with PyAV (bundled ffmpeg) and re-encodes it to 16 kHz mono FLAC --
lossless, libsndfile-native, on Whisper's accepted-format list, and far smaller
than WAV -- so every upload reaches the endpoint in a form it can always decode.

Unlike the fail-open VAD guard (:func:`nextext.core.vad.has_speech`),
normalization is **fail-closed**: an undecodable input raises
:class:`AudioDecodeError` rather than letting unusable bytes reach the endpoint.
"""

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import av
from av.error import FFmpegError
from loguru import logger

__all__ = ["AudioDecodeError", "normalize_for_transcription"]

TARGET_SAMPLE_RATE = 16000
TARGET_LAYOUT = "mono"
TARGET_SAMPLE_FORMAT = "s16"


class AudioDecodeError(ValueError):
    """Raised when an upload cannot be decoded into Whisper-ready audio."""


def _transcode_to_flac(src: Path, dst: Path) -> None:
    """Decode ``src`` and write a 16 kHz mono FLAC to ``dst``.

    Args:
        src (Path): Source media in any container/codec PyAV can decode.
        dst (Path): Destination path for the 16 kHz mono FLAC output.

    Raises:
        AudioDecodeError: If ``src`` has no audio stream or cannot be
            decoded/encoded.
    """
    resampler = av.AudioResampler(
        format=TARGET_SAMPLE_FORMAT,
        layout=TARGET_LAYOUT,
        rate=TARGET_SAMPLE_RATE,
    )
    try:
        with av.open(str(src)) as in_container, av.open(str(dst), mode="w") as out_container:
            if not in_container.streams.audio:
                raise AudioDecodeError(f"Could not decode audio file '{src.name}': no audio stream found.")
            in_stream = in_container.streams.audio[0]
            out_stream = out_container.add_stream("flac", rate=TARGET_SAMPLE_RATE)
            out_stream.codec_context.format = TARGET_SAMPLE_FORMAT
            out_stream.codec_context.layout = TARGET_LAYOUT

            for frame in in_container.decode(in_stream):
                for resampled in resampler.resample(frame):
                    resampled.pts = None
                    for packet in out_stream.encode(resampled):
                        out_container.mux(packet)
            for resampled in resampler.resample(None):  # flush the resampler
                resampled.pts = None
                for packet in out_stream.encode(resampled):
                    out_container.mux(packet)
            for packet in out_stream.encode(None):  # flush the encoder
                out_container.mux(packet)
    except AudioDecodeError:
        raise
    except (FFmpegError, ValueError) as exc:
        raise AudioDecodeError(
            f"Could not decode audio file '{src.name}'; it may be corrupt or in an unsupported format."
        ) from exc


@contextmanager
def normalize_for_transcription(file_path: Path) -> Iterator[Path]:
    """Yield a 16 kHz mono FLAC re-encode of ``file_path``; delete it on exit.

    Decodes ``file_path`` with PyAV regardless of its container/codec and
    re-encodes it to a temporary 16 kHz mono FLAC that the soundfile-based
    Whisper loader can always read. The temporary file is removed when the
    context exits, whether or not the body raised.

    Args:
        file_path (Path): Path to the source upload (any decodable format).

    Yields:
        Path: Path to the temporary 16 kHz mono FLAC file.

    Raises:
        AudioDecodeError: If ``file_path`` cannot be decoded.
    """
    fd, tmp_name = tempfile.mkstemp(suffix=".flac", prefix=f"{file_path.stem}-")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        _transcode_to_flac(file_path, tmp_path)
        logger.info(
            "Normalized '{}' to 16 kHz mono FLAC ({} bytes).",
            file_path.name,
            tmp_path.stat().st_size,
        )
        yield tmp_path
    finally:
        tmp_path.unlink(missing_ok=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
uv run pytest tests/test_audio.py -q
```
Expected: both tests PASS.

If `test_normalize_produces_16k_mono_flac` errors inside `_transcode_to_flac`
with an encoder complaint about channel layout or sample format, the
empirical PyAV encoder check applies (see Task 5, Step 2): keep the
`out_stream.codec_context.format`/`.layout` string assignments; if PyAV's typed
stubs reject the `str` on assignment, wrap them as `av.AudioFormat(TARGET_SAMPLE_FORMAT)`
and `av.AudioLayout(TARGET_LAYOUT)`.

- [ ] **Step 5: Lint/type-check the new module**

Run:
```bash
uv run ruff check nextext/core/audio.py tests/test_audio.py
uv run mypy . --exclude="build/|legacy/"
```
Expected: clean. If mypy flags the two `codec_context` assignments as
`[assignment]`, switch them to `av.AudioFormat(...)` / `av.AudioLayout(...)`
(do **not** add a blanket `# type: ignore` — strict mode has `warn_unused_ignores`,
so an unneeded ignore fails the gate).

- [ ] **Step 6: Commit**

```bash
git add nextext/core/audio.py tests/test_audio.py
git commit -m "feat: add audio-normalization agent (16 kHz mono FLAC via PyAV)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Wire normalization into transcription()

**Files:**
- Modify: `nextext/core/transcription.py:19` (imports) and `:330-377` (the request block)
- Test: `tests/test_transcription.py`

- [ ] **Step 1: Write/adjust the tests first**

In `tests/test_transcription.py`:

(a) Add these imports near the top (alongside the existing `from types import SimpleNamespace` / `from unittest.mock import MagicMock`):

```python
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path

from nextext.core import audio
```

(b) Add a module-level passthrough helper:

```python
@contextmanager
def _passthrough_normalize(file_path: Path) -> Iterator[Path]:
    """Stand-in for normalize_for_transcription that yields the input unchanged.

    Args:
        file_path (Path): The path passed through verbatim.

    Yields:
        Path: ``file_path`` unchanged.
    """
    yield file_path
```

(c) In **both** existing tests that call `transcriber.transcription()` —
`test_external_transcriber_transcription_populates_src_lang` and
`test_external_transcriber_normalizes_full_language_name` — add this line right
after the existing `monkeypatch.setattr(transcription, "has_speech", lambda _path: True)`:

```python
    monkeypatch.setattr(transcription, "normalize_for_transcription", _passthrough_normalize)
```

(d) Add two new tests at the end of the file:

```python
def test_transcription_normalizes_before_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """transcription() routes the upload through normalize_for_transcription.

    Args:
        monkeypatch (pytest.MonkeyPatch): Patches the client, VAD guard, and
            the normalization seam.
    """
    seg = SimpleNamespace(start=0.0, end=1.0, text="Hi.", no_speech_prob=0.1)
    fake_response = SimpleNamespace(segments=[seg], language="en")
    fake_client = MagicMock()
    fake_client.audio.transcriptions.create.return_value = fake_response

    transcriber = ExternalWhisperTranscriber.__new__(ExternalWhisperTranscriber)
    transcriber.file_path = transcription.Path(__file__)
    transcriber.src_lang = "en"
    transcriber.task = "transcribe"
    transcriber._model_id = "whisper-1"
    transcriber._client = None
    transcriber.transcription_result = None

    monkeypatch.setattr(type(transcriber), "_get_client", property(lambda self: fake_client))
    monkeypatch.setattr(transcription, "has_speech", lambda _path: True)

    seen: list[Path] = []

    @contextmanager
    def _spy(file_path: Path) -> Iterator[Path]:
        seen.append(file_path)
        yield file_path

    monkeypatch.setattr(transcription, "normalize_for_transcription", _spy)

    transcriber.transcription()

    assert seen == [transcriber.file_path]
    fake_client.audio.transcriptions.create.assert_called_once()


def test_transcription_raises_on_undecodable_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An AudioDecodeError from normalization aborts before any API call.

    Args:
        monkeypatch (pytest.MonkeyPatch): Patches the client, VAD guard, and
            the normalization seam to raise.
    """
    fake_client = MagicMock()
    transcriber = ExternalWhisperTranscriber.__new__(ExternalWhisperTranscriber)
    transcriber.file_path = transcription.Path("voice.ogg")
    transcriber.src_lang = None
    transcriber.task = "transcribe"
    transcriber._model_id = "whisper-1"
    transcriber._client = None
    transcriber.transcription_result = None

    monkeypatch.setattr(type(transcriber), "_get_client", property(lambda self: fake_client))
    monkeypatch.setattr(transcription, "has_speech", lambda _path: True)

    @contextmanager
    def _raise(file_path: Path) -> Iterator[Path]:
        raise audio.AudioDecodeError(f"Could not decode '{file_path.name}'.")
        yield file_path  # pragma: no cover - unreachable, satisfies generator typing

    monkeypatch.setattr(transcription, "normalize_for_transcription", _raise)

    with pytest.raises(audio.AudioDecodeError):
        transcriber.transcription()

    fake_client.audio.transcriptions.create.assert_not_called()
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:
```bash
uv run pytest tests/test_transcription.py -q
```
Expected: the two new tests FAIL — `test_transcription_normalizes_before_upload`
fails its `assert seen == [...]` (the normalizer is not called yet) and
`test_transcription_raises_on_undecodable_audio` fails its `pytest.raises`
(no exception yet). The two edited existing tests still PASS (they reference the
not-yet-used `normalize_for_transcription` only via `monkeypatch.setattr`, which
will error until the symbol exists — if so, that's expected red for this step).

- [ ] **Step 3: Implement the wiring**

In `nextext/core/transcription.py`, add the import after line 17–19 (with the
other `from nextext.core...` imports):

```python
from nextext.core.audio import normalize_for_transcription
from nextext.core.diarization import diarize_file
from nextext.core.vad import has_speech
```

Then replace the request block (current `:330-377`, from
`client = self._get_client` through the `except APIStatusError ... raise`) with:

```python
        provider = load_inference_env().provider
        with normalize_for_transcription(self.file_path) as audio_path:
            file_size = audio_path.stat().st_size
            logger.info(
                "External Whisper request: model='{}' task='{}' language='{}' "
                "file='{}' (normalized from '{}') size={}B",
                self._model_id,
                self.task,
                self.src_lang,
                audio_path.name,
                self.file_path.name,
                file_size,
            )
            if provider == "openai" and file_size > OPENAI_WHISPER_MAX_UPLOAD_BYTES:
                size_mb = file_size / (1024 * 1024)
                limit_mb = OPENAI_WHISPER_MAX_UPLOAD_BYTES / (1024 * 1024)
                raise ValueError(
                    f"Audio file '{self.file_path.name}' is {size_mb:.1f} MB after "
                    f"normalization, which exceeds OpenAI's {limit_mb:.0f} MB Whisper "
                    "upload limit. Compress or split the file before retrying, or point "
                    "WHISPER_API_BASE at a self-hosted endpoint without a hard cap "
                    "(e.g. the vllm-service audio container)."
                )
            client = self._get_client
            try:
                with open(audio_path, "rb") as f:
                    if self.task == "translate":
                        response = client.audio.translations.create(
                            model=self._model_id,
                            file=f,
                            response_format="verbose_json",
                        )
                    else:
                        kwargs: dict[str, Any] = {
                            "model": self._model_id,
                            "file": f,
                            "response_format": "verbose_json",
                            "timestamp_granularities": ["segment"],
                        }
                        if self.src_lang:
                            kwargs["language"] = self.src_lang
                        response = client.audio.transcriptions.create(**kwargs)
            except APIStatusError as exc:
                body = getattr(exc, "response", None)
                body_text = body.text if body is not None else ""
                logger.error(
                    "External Whisper API error {}: {}",
                    exc.status_code,
                    body_text or exc.message,
                )
                raise
```

The `raw_segments = [...]` block that follows (current `:378+`) is unchanged and
stays at method-body indentation, after the `with` block — `response` is bound
inside the block and read after it. VAD still runs first on the original file
(unchanged early-return path above), and `diarization()` still uploads the
original file (no change).

- [ ] **Step 4: Run the transcription tests to verify they pass**

Run:
```bash
uv run pytest tests/test_transcription.py -q
```
Expected: all tests PASS (the two new ones plus the two edited ones plus the rest).

- [ ] **Step 5: Commit**

```bash
git add nextext/core/transcription.py tests/test_transcription.py
git commit -m "fix: normalize audio to 16 kHz mono FLAC before the Whisper upload

Fixes HTTP 400 'Invalid or unsupported audio file' on Ogg Opus and other
libsndfile-incompatible uploads; undecodable input now fails fast with a
clear AudioDecodeError instead of a cryptic upstream 400.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Update documentation and the Dockerfile comment

**Files:**
- Modify: `AGENTS.md:35-40`
- Modify: `CLAUDE.md` (Project Overview, Pipeline flow step 1, Key modules, Docker)
- Modify: `docker/Dockerfile.backend:42-43`

- [ ] **Step 1: Update AGENTS.md**

In the "Transcription & Diarization Agent" section:

Replace the **Key files** line (`:35`):
```markdown
- **Key files:** `nextext/core/transcription.py`, `nextext/core/audio.py`, `nextext/core/vad.py`, `nextext/core/diarization.py`, `transcription_pipeline()` (`nextext/pipeline.py`).
```

Replace the **Responsibilities** line (`:36`):
```markdown
- **Responsibilities:** Screen the file with the external `/vad` speech guard, normalize it to 16 kHz mono FLAC (`nextext/core/audio.py`, PyAV) so libsndfile-only Whisper servers can decode any source format, forward it to an OpenAI-compatible `/v1/audio/transcriptions` endpoint, optionally label speakers via the out-of-process `/diarize` HTTP service (`n_speakers > 1`), and emit a normalized DataFrame used by every downstream agent.
```

Replace the **Dependencies** line (`:40`):
```markdown
- **Dependencies:** `openai` SDK, `httpx`, and `av` (PyAV — bundled ffmpeg) for the pre-upload audio normalization. No torch, no GPU; PyAV's wheel is the only local media dependency (no apt audio tooling). Normalization is fail-closed: an undecodable upload raises `AudioDecodeError`.
```

- [ ] **Step 2: Update CLAUDE.md**

Replace in the Project Overview paragraph:
```
Only spaCy/NLTK word-level NLP runs in-process — the backend ships no model weights, no local audio tooling, and needs no GPU.
```
with:
```
Only spaCy/NLTK word-level NLP runs in-process; every upload is re-encoded to 16 kHz mono FLAC via the PyAV wheel (bundled ffmpeg) before transcription. The backend ships no model weights and needs no GPU — PyAV is the only local media dependency, and no apt audio tooling is installed.
```

Replace the Pipeline-flow step 1 opening:
```
1. **Transcription** (always-on) → external Whisper API (`/v1/audio/transcriptions`, always in the source language) behind an external `/vad` speech guard
```
with:
```
1. **Transcription** (always-on) → every upload is re-encoded to 16 kHz mono FLAC (`nextext/core/audio.py`, PyAV) so libsndfile-only Whisper servers can decode it → external Whisper API (`/v1/audio/transcriptions`, always in the source language) behind an external `/vad` speech guard
```

Add to the "Key modules" list, immediately after the
`nextext/core/transcription.py` bullet:
```
- `nextext/core/audio.py` — audio-normalization agent: re-encodes any upload to 16 kHz mono FLAC via PyAV (bundled ffmpeg) before the Whisper call; fail-closed (`AudioDecodeError`) on undecodable input.
```

Replace in the Docker section, the backend bullet fragment:
```
multi-stage `uv` build (no extras; runtime apt is `curl` only — all inference, including the VAD guard, is external).
```
with:
```
multi-stage `uv` build (no extras; runtime apt is `curl` only — all inference, including the VAD guard, is external; audio normalization uses the PyAV wheel, so no apt audio tooling is added).
```

- [ ] **Step 3: Update the Dockerfile comment**

In `docker/Dockerfile.backend`, replace the runtime comment (`:42-43`):
```dockerfile
# curl for the healthcheck only; all model inference (including the VAD
# speech guard) runs on external endpoints, so no local audio tooling is needed.
```
with:
```dockerfile
# curl for the healthcheck only; all model inference (including the VAD speech
# guard) runs on external endpoints. Audio is normalized to FLAC in-process via
# the PyAV wheel (bundled ffmpeg), so no apt audio tooling is installed here.
```

- [ ] **Step 4: Commit**

```bash
git add AGENTS.md CLAUDE.md docker/Dockerfile.backend
git commit -m "docs: document client-side audio normalization (PyAV)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Full verification

**Files:** none (verification only; commit any formatter changes)

- [ ] **Step 1: Run the full test suite**

Run:
```bash
uv run pytest -q
```
Expected: all tests PASS. Report the pass/fail counts (per `CLAUDE.md`).

- [ ] **Step 2: Run all pre-commit gates**

Run:
```bash
uv run pre-commit run --all-files
```
Expected: ruff-check, ruff-format, and mypy (strict, over `.`) all pass.

PyAV-typing fixups, if mypy reports them:
- `[assignment]` on `out_stream.codec_context.format`/`.layout` → wrap the
  values as `av.AudioFormat(TARGET_SAMPLE_FORMAT)` / `av.AudioLayout(TARGET_LAYOUT)`.
- If `from av.error import FFmpegError` is flagged as unresolved, use
  `av.FFmpegError` instead (import surface differs slightly across PyAV releases).
- Do not add blanket `# type: ignore`; strict mode's `warn_unused_ignores` will
  fail an unused one. Re-run until clean.

- [ ] **Step 3: Commit any formatter/lint changes**

```bash
git add -A
git commit -m "chore: apply formatter/lint fixups for audio normalization

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
(Skip if nothing changed.)

---

## Self-Review Notes

- **Spec coverage:** new agent (Task 2) ✓; insertion point with VAD-first ordering, size log/cap moved onto the normalized file, both task branches wrapped (Task 3) ✓; fail-closed `AudioDecodeError` (Tasks 2–3) ✓; `av>=12` dep, no torch, no apt (Task 1) ✓; tests incl. the monkeypatch seam (Tasks 2–3) ✓; AGENTS.md/CLAUDE.md/Dockerfile updates (Task 4) ✓; chunking left out of scope ✓.
- **Deliberate deviation from spec:** the spec proposed synthesizing an *Ogg Opus* fixture; the plan synthesizes a stereo 44.1 kHz *WAV* with the stdlib `wave` module instead. Rationale: the normalizer decodes every format through the same PyAV path, so a WAV source still exercises the real logic (resample + downmix + FLAC encode) while avoiding encoder-availability and PyAV-typing flakiness in the fixture. The "libsndfile can't read it" property belongs to PyAV/ffmpeg, which we trust. Behavior and coverage are unchanged.
- **Type consistency:** `normalize_for_transcription(file_path: Path) -> Iterator[Path]`, `AudioDecodeError(ValueError)`, and the module constants are referenced identically across `audio.py`, `transcription.py`, and both test files.
- **Empirical gate:** PyAV's exact typed-stub posture under strict mypy can't be known until `av` is installed (Task 1); Task 2 Step 5 and Task 5 Step 2 carry the precise fixups, so this is a known, bounded follow-up, not a placeholder.
