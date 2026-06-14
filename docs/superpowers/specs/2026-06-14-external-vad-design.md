# External VAD speech guard — design

**Date:** 2026-06-14
**Status:** Approved
**Branch:** `refactor/external-inference-unified-image` (PR #34 — this lands on top of it)

> **Amendment 2026-06-14 (supersedes the unset = disabled semantics below — PR #37):**
> VAD is no longer dedicated-only. `load_vad_env()` resolves `VAD_API_BASE` three
> ways: a URL is used as-is; an explicit off token (`off`/`false`/`no`/`0`) switches
> the guard off; **unset falls back to the central `OPENAI_API_BASE`** with one
> trailing `/v1` stripped — so the guard is **on by default** and posts to
> `{central-root}/vad`. NER and diarization gained the same central fallback. The
> sections below have been updated to match where they stated ongoing behavior; the
> fail-open response to errors/malformed payloads is unchanged, and the backend
> makes no claim about which `/vad` implementation the endpoint provides.

## Problem

Nextext routes Whisper, NER (`/gliner`), and diarization (`/diarize`) to external
HTTP services, but voice-activity detection (VAD) still runs locally: a bundled
Silero ONNX model (`pysilero-vad`) plus an RMS energy check, both on a
locally-ffmpeg-decoded waveform, gated by `VAD_ENABLED`. This keeps an ONNX model
and audio-DSP code in the otherwise model-free backend.

Goal: make VAD an external service exactly like NER and diarization, and remove
**all** local VAD/audio-DSP code (RMS, Silero, ffmpeg/numpy decode). After this
change the backend ships zero model weights and no audio signal processing.

## Approach (decided)

- **Replace everything** — drop the local RMS check, the Silero ONNX VAD, and the
  ffmpeg/numpy waveform decode. A `/vad` service (file upload, like `/diarize`)
  becomes the sole pre-Whisper speech guard. `pysilero-vad` is removed entirely.
- **Land on PR #34's branch** — committed onto `refactor/external-inference-unified-image`
  as small topical commits, so the one PR carries the complete external-inference move.

## Component: `nextext/core/vad.py`

New module mirroring `nextext/core/diarization.py` (both upload an audio file).

```python
has_speech(file_path: Path) -> bool
```

- Uploads the file (multipart `file=<audio>`) to `{VAD_API_BASE}/vad`.
- `Authorization: Bearer <OPENAI_API_KEY>` when set (token reused, as NER/diar do).
- Module-level `httpx.post` so tests monkeypatch `vad.httpx`.
- The module docstring carries the `POST /vad` contract (as `diarization.py`
  documents `/diarize`).

### Wire contract

`POST {VAD_API_BASE}/vad`
- multipart/form-data: `file` — the audio bytes (any ffmpeg-decodable format; the
  server decodes/resamples itself — no local DSP)
- `Authorization: Bearer <token>` (optional)

`200 OK` → `{"has_speech": true|false}`

## Fail/disabled semantics — fail-open guard (the key nuance)

VAD is a **guard**, not a data producer. NER/diarization degrade to *less data*
(empty entities / no labels); a wrong VAD "no speech" would **drop the whole
transcription**. So the client is fail-open:

| Situation | `has_speech()` returns | Effect |
|---|---|---|
| `VAD_API_BASE=off` (or `false`/`no`/`0`) | `True` | guard switched off → transcribe everything |
| `VAD_API_BASE` and `OPENAI_API_BASE` both unset | `True` | no endpoint resolves → transcribe everything |
| service unreachable / HTTP error / malformed payload | `True` (+ warning log) | transcribe anyway — an outage never drops audio |
| service returns `{"has_speech": false}` | `False` | the only case the Whisper upload is skipped |

## `nextext/utils/env_cfg.py`

- `VadConfig` changes from `{enabled: bool}` → `{api_base: str, api_key: str, timeout: float}`
  (identical shape to `NerConfig` / `DiarizationConfig`).
- `load_vad_env()` resolves `VAD_API_BASE` three ways — a URL (strip + rstrip `/`);
  an off token (`off`/`false`/`no`/`0`) → empty; unset → `_central_endpoint_root()`
  (`OPENAI_API_BASE` with one trailing `/v1` stripped) — plus `OPENAI_API_KEY`
  (reused token) and `VAD_TIMEOUT` (new `DEFAULT_VAD_TIMEOUT = 60.0`; non-numeric
  / non-positive values warn and fall back, as NER/diar do).
- `VAD_ENABLED` is removed — `VAD_API_BASE` is the on/off switch (`=off` disables;
  unset uses the central endpoint).

## `nextext/core/transcription.py`

Remove all local audio code: `_get_vad`, `_detect_speech_vad`,
`_load_audio_waveform`, `_audio_has_speech`, the `SILENCE_RMS_THRESHOLD` /
`VAD_SPEECH_THRESHOLD` / `_vad_cache` symbols, and the now-unused
`numpy` / `subprocess` / `load_vad_env` imports.

Keep `NO_SPEECH_THRESHOLD` + `_filter_no_speech_segments` (post-transcription
hallucination filter — unrelated to VAD).

In `transcription()` the guard becomes:

```python
if not has_speech(self.file_path):
    self.transcription_result = {"segments": []}
    return
```

Module docstring updated to describe the external `/vad` guard.

## Dependencies

Remove `pysilero-vad>=2.1.1,<3` from `pyproject.toml` (onnxruntime drops
transitively). Regenerate `uv.lock`.

## Tests

- **New `tests/test_vad.py`** (mirrors `test_diarization.py`): respx-mocked `/vad`
  → speech / no-speech; unset base → `True`; transport / HTTP / malformed → `True`
  (fail-open); bearer header present and absent.
- **`tests/test_transcription.py`**: delete the local-VAD tests
  (`_audio_has_speech` ×4, `_detect_speech_vad` ×4, `_get_vad`, real-ONNX inference,
  `_load_audio_waveform` ×3); rewire the transcriber skip/pass tests to monkeypatch
  `transcription.has_speech`.
- **`tests/test_env_cfg.py`**: `load_vad_env` tests for `VAD_API_BASE` / `VAD_TIMEOUT`
  modeled on the `load_ner_env` tests.

## Docs

`CLAUDE.md`, `README.md`, `AGENTS.md`, `.env.example`: VAD joins NER/diarization as
an external service that defaults to the central endpoint; drop "ONNX Silero VAD
runs in-process", the pysilero pin note, and `VAD_ENABLED`; add `VAD_API_BASE`
(incl. the `=off` switch) / `VAD_TIMEOUT`. The docs make no claim about which
`/vad` implementation the endpoint provides.

## Accepted consequence (amended 2026-06-14, PR #37)

Originally `VAD_API_BASE` was unset-by-default and the pre-filter was off out of the
box. As of PR #37 the guard **falls back to the central endpoint**, so it runs ahead
of every transcription by default; when that endpoint serves no `/vad` the fail-open
client logs and transcribes anyway — one extra round-trip per file. Set
`VAD_API_BASE=off` to skip the guard entirely.

## Out of scope

- Implementing the `/vad` server (separate repo; the contract is documented in
  `vad.py` for that follow-up).
- Per-job VAD toggle — VAD stays a backend-env-only guard.
