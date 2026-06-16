# Client-side audio normalization before transcription

- **Date:** 2026-06-16
- **Status:** Approved (design)
- **Topic:** Normalize uploads to 16 kHz mono FLAC before the external Whisper call

## Problem

Transcription of some uploads (reproduced with a `.ogg` file) fails with HTTP 400
from the inference stack:

```
soundfile.LibsndfileError: Unspecified internal error.
ValueError: Invalid or unsupported audio file.   # vLLM
litellm.BadRequestError: OpenAIException - Invalid or unsupported audio file.
```

### Root cause

Nextext forwards the raw uploaded bytes unchanged to the Whisper endpoint
(`nextext/core/transcription.py:352-368` opens the file `"rb"` and streams it to
`client.audio.transcriptions.create`). The vLLM ASR server decodes audio with
**`soundfile`/libsndfile only** — its traceback goes
`load_audio` → `load_audio_soundfile` → `soundfile.read`, with **no ffmpeg
fallback**. libsndfile cannot decode the uploaded file (the `.ogg` is almost
certainly **Ogg Opus**, the default for phone/web/messenger voice recordings),
so it rejects the request. The `libmpg123` "Illegal Audio-MPEG-Header / resync"
spam in the server log is libsndfile's MP3 fallback flailing on non-MP3 bytes —
a path it only reaches because it already declined the file as Ogg/Opus.

This is not an isolated `.ogg` problem. The frontend advertises seven upload
types (`nextext/frontend/app.py:540`): `mp3, m4a, mp4, mkv, ogg, wav, webm`. A
soundfile-only decoder reliably handles only `wav`, `mp3`, and Ogg/Vorbis — it
cannot read the `m4a`, `mp4`, `mkv`, or `webm` containers at all. The decode gap
is structural; `.ogg`/Opus is just the case that surfaced it.

By contrast the `/vad` and `/diarize` services decode server-side with ffmpeg and
accept any format (`nextext/core/vad.py:21-22`), which is why VAD passed the file
through and only the Whisper hop failed.

## Goals

- Every upload reaches the Whisper endpoint in a container/codec the
  soundfile-based loader can always decode.
- Fix applies to **both** entry paths (API job worker and `nextext-cli`), which
  both funnel through `transcription_pipeline` → `ExternalWhisperTranscriber.transcription()`.
- Large/long inputs are not made worse (avoid uncompressed blow-up).
- Genuinely undecodable (corrupt/truncated) inputs fail fast with a clear message
  instead of a cryptic upstream 400.
- Keep the no-torch invariant (`tests/test_no_torch.py`) and add no apt audio
  tooling to the images.

## Non-goals (explicit follow-ups)

- **Chunking / duration limits** for multi-hour files. This change fixes decode
  and curbs upload size, not length. A long-but-decodable file may still strain
  a single Whisper request; chunking is a separate piece of work.
- Changing VAD or diarization (they already accept any format).
- Re-encoding for the VAD/diarize uploads (unnecessary — they decode server-side).

## Approach (selected)

A dedicated stateless agent normalizes the audio; `transcription()` wraps its
upload with it. Both entry paths inherit the fix because both call
`transcription_pipeline`.

Alternatives rejected:

- **Transcode at the `pipeline.py` boundary** — would also re-encode for
  VAD/diarize (needless work) and split responsibility across modules.
- **Lazy fallback (send original, catch 400, transcode, retry)** — brittle 400
  classification, wasted round-trip, no deterministic size control.

## Components

### New: `nextext/core/audio.py`

A stateless "audio normalization" agent with a single public entry point:

```python
@contextmanager
def normalize_for_transcription(file_path: Path) -> Iterator[Path]:
    """Yield a 16 kHz mono FLAC re-encode of file_path; delete it on exit."""
```

- **Input:** path to any user upload (any container/codec).
- **Output:** path to a temporary `.flac` file, 16 kHz, mono, deleted when the
  context exits (success or error).
- **Dependency:** PyAV (`av`).

**Algorithm:**

1. `av.open(str(file_path))`; select the first audio stream. No audio stream →
   raise `AudioDecodeError`.
2. Write to a temp file created with suffix `.flac` (via `tempfile`); open an
   output container, add a FLAC stream at `rate=16000`, mono.
3. Resample with `av.AudioResampler(format="s16", layout="mono", rate=16000)`;
   decode → resample → encode → mux; then flush the resampler and the encoder.
   (PyAV's `resampler.resample(frame)` returns a *list* of frames — iterate it.)
4. Close both containers.
5. Any PyAV decode/encode error (`av.error.*` / `av.AVError`), or an input with
   no audio stream, is re-raised as `AudioDecodeError`.

**Target format rationale:** FLAC is lossless, libsndfile-native (so the vLLM
soundfile loader decodes it cleanly), on Whisper/OpenAI's accepted-format list,
and roughly an order of magnitude smaller than WAV — so long recordings do not
balloon. 16 kHz mono is exactly Whisper's internal input, so the server does no
extra resampling and the upload is minimal.

**Always normalize** (not gated by extension): deterministic, and extensions
lie. An already-16 kHz-mono file just gets a cheap re-encode.

**Errors:** define `class AudioDecodeError(ValueError)` in the same module.
Fail-closed (the opposite of VAD's fail-open): we cannot send garbage to the
endpoint, so an undecodable input must surface as a clear error, e.g.
*"Could not decode audio file 'x.ogg'; it may be corrupt or in an unsupported
format."* This also catches genuinely corrupt files early.

### Changed: `nextext/core/transcription.py`

- Import at module level: `from nextext.core.audio import normalize_for_transcription`
  (module-level so tests can monkeypatch `transcription.normalize_for_transcription`,
  mirroring how `has_speech`/`Path` are patched today).
- In `transcription()` (`:303-395`), keep the ordering for efficiency:
  1. `has_speech(self.file_path)` runs **first on the original** — silent files
     skip before any transcode work.
  2. `with normalize_for_transcription(self.file_path) as audio_path:` wraps the
     upload, covering **both** the `transcribe` and `translate` branches.
  3. Move the upload-size log line and the `OPENAI_WHISPER_MAX_UPLOAD_BYTES`
     check onto `audio_path` (the bytes actually uploaded), keeping the
     `provider == "openai"` gate. The log keeps the original filename for
     traceability, e.g. `file='<tmp>.flac' (from 'voice.ogg') size=...B`.
  4. `open(audio_path, "rb")` for the request; the `.flac` suffix gives the
     endpoint a correct extension for format detection.
- `diarization()` is unchanged — it keeps uploading the original file.

## Data flow

```
upload (any format)
  └─ transcription()
       ├─ has_speech(original)         # VAD, fail-open, server decodes any format
       │     └─ no speech → skip (unchanged)
       └─ normalize_for_transcription(original)   # PyAV → 16 kHz mono FLAC (temp)
             ├─ decode error → AudioDecodeError → job fails with clear reason
             └─ audio_path (.flac)
                   ├─ [openai] size cap check on audio_path
                   └─ client.audio.transcriptions.create(file=audio_path)  → segments
       (temp .flac deleted on context exit)

diarization(): uploads original file (unchanged)
```

## Dependency changes

- Add `av>=12` (PyAV) to `[project].dependencies` in `pyproject.toml` (modern
  floor; see Risks). PyAV ships a wheel that bundles ffmpeg, so:
  - no apt packages added to `docker/Dockerfile.backend`,
  - local `nextext-cli` users need no system ffmpeg,
  - no torch pulled in — `tests/test_no_torch.py` stays green.
- `uv.lock` regenerated via `uv lock`.

## Testing

New `tests/test_audio.py`:

- **Round-trip:** synthesize a tiny Ogg Opus sample with PyAV at test time
  (no binary fixtures, fully offline), run `normalize_for_transcription`, and
  assert the yielded file exists, is FLAC, decodes to 16 kHz mono, and is
  removed after the context exits.
- **Failure:** point the function at non-audio bytes (e.g. a small text file)
  and assert it raises `AudioDecodeError`.

Update `tests/test_transcription.py`:

- The existing `transcription()` tests set `file_path` to
  `transcription.Path(__file__)` (a `.py` file) and stub `has_speech`. They must
  also stub the new seam, e.g.
  `monkeypatch.setattr(transcription, "normalize_for_transcription", _passthrough)`,
  where `_passthrough` is a `@contextmanager` yielding the original path. Add a
  shared helper rather than repeating it.
- Add a test asserting `transcription()` raises (and makes no client call) when
  `normalize_for_transcription` raises `AudioDecodeError`.

Full-suite gates (per `CLAUDE.md`): `uv run pytest`, then
`pre-commit run --all-files` (ruff, mypy strict, docstrings). `av` has no type
stubs; `[tool.mypy] ignore_missing_imports = true` already covers it.

## Documentation updates

- `AGENTS.md` — add the audio-normalization agent (I/O contract: path in → temp
  16 kHz mono FLAC path out; fail-closed).
- `CLAUDE.md` — update the "no local audio tooling" claims in the Project
  Overview / Context and the `core/` module list to: audio is normalized
  client-side via the PyAV wheel before transcription (the one local media
  dependency); still no GPU, no model weights, no apt audio tooling.
- `docker/Dockerfile.backend:42-43` — revise the "no local audio tooling is
  needed" comment to reflect PyAV-based normalization (wheel, not apt).

## Risks

- **PyAV API drift** across versions (notably `AudioResampler.resample` returning
  a list). Mitigated by pinning a modern floor (`av>=12`) and covering the
  round-trip in tests.
- **FLAC acceptance** by the endpoint: FLAC is on the OpenAI/Whisper accepted
  list and is libsndfile-native, so the soundfile loader handles it; low risk.
- **Wheel size / build:** PyAV wheels are prebuilt with bundled ffmpeg; no
  builder apt changes expected.
