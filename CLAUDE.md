# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Data confidentiality — hard rule

**NEVER expose actual production or testing data in any file committed or
pushed to git.** This covers not only file contents but also metadata that
references real data: filenames, file descriptions, social-media account
names or handles, user identifications, sample records, log excerpts, and
screenshots. It applies everywhere git sees — source code, tests, fixtures,
docs, examples, configs, commit messages, and CI files. Use fully synthetic,
invented placeholders instead.

**Likewise, NEVER expose local filepaths from development machines** —
absolute paths or home directories such as `/Users/<name>/...`,
`/home/<name>/...`, or `C:\Users\...` — anywhere git sees. The only
permitted paths are relative project paths starting from the project's
root (e.g. `docker/compose.yaml`).

## Project Overview

Nextext is a modular audio analysis toolkit that transcribes, translates, and analyzes natural language from audio/video files. All model inference runs on external endpoints: Whisper transcription via an OpenAI-compatible audio API, voice-activity detection (`/vad`), speaker diarization (`/diarize`), and GLiNER NER (`/gliner`) via dedicated out-of-process HTTP services, and LLMs (Ollama, vLLM, or OpenAI-compatible endpoints) for translation, summarization, and hate-speech detection. Only spaCy/NLTK word-level NLP runs in-process; every upload is re-encoded to 16 kHz mono FLAC via the PyAV wheel (bundled ffmpeg) before transcription. The backend ships no model weights and needs no GPU — PyAV is the only local media dependency, and no apt audio tooling is installed.

## Project Context

- Local ML runtimes (openai-whisper, pyannote, GLiNER, Silero VAD, torch) have been removed; every model call is an HTTP request to an external endpoint. `tests/test_no_torch.py` pins the no-torch invariant — camel-tools' torch/transformers requirements are excluded via `[tool.uv] override-dependencies`.
- Docker base image is the pinned `ghcr.io/astral-sh/uv:*-python3.12-trixie-slim` across backend and frontend Dockerfiles.

## Commands

```bash
# Install dependencies
uv sync                    # production deps
uv sync --group dev        # include dev deps (pytest, ruff, pyrefly, pre-commit)

# Run the app
uv run nextext-api         # FastAPI backend on port 8000
uv run nextext-cli -f <file> [args]  # CLI mode (in-process, no backend required)

# Frontend (React SPA) — run via Docker or the Vite dev server
make build && make up-dev  # Docker: build images, then run detached; publishes on NEXTEXT_HOST_PORT (or: make dev)
# cd frontend && pnpm dev  # local Vite dev server (proxies /api/v1 to localhost:8000)

# Preload spaCy/NLTK language resources (the only local downloads)
NEXTEXT_OFFLINE=0 uv run load-models

# Tests
uv run pytest              # run full test suite
uv run pytest tests/test_pipeline.py  # single test file
uv run pytest -k "test_name"          # single test by name

# Linting & formatting (also enforced by pre-commit hooks)
ruff check --fix           # lint with auto-fix
ruff format                # format code
uv run pyrefly check
```

## Testing

- Always run the full test suite (`pytest`) after making changes and report pass/fail counts.
- When tests fail, fix the root cause rather than patching tests to match stale/removed code.
- Verify with `pre-commit run --all-files` (pyrefly, lint, docstrings) before declaring work complete.

Tests are in `tests/` using pytest with monkeypatch fixtures and `respx` for mocking the HTTP inference clients (Whisper, NER, diarization). Tests simulate Docker detection and environment configuration. No GPU, no network, and no model downloads required for tests.

Several modules call `load_dotenv()` at import time (`nextext/utils/env_cfg.py`, `nextext/core/openai_cfg.py`, `nextext/core/words.py`, `nextext/utils/model_loader.py`), which copies a local, uncommitted `.env`'s values straight into the real process environment the first time one of those modules is imported — not into a test-scoped sandbox, and not reverted by `monkeypatch`. A developer `.env` with e.g. `RESPONSE_LANGUAGE=de` (a normal local-dev setup, see `.env.example`) then silently outranks any test asserting "unset" default behavior, for the rest of that pytest session. `tests/conftest.py` carries an autouse fixture that clears `RESPONSE_LANGUAGE`/`NEXTEXT_RESPONSE_LANGUAGE` before every test so the suite is hermetic against ambient `.env` state; add the same pattern there for any other env var a test needs to assume is unset by default.

## Docstrings & Style

- All new/modified Python functions must have Google-style docstrings.
- Python 3.12 is the target; prefer explicit types and distinct variable names across branches to satisfy pyrefly.

## Architecture

**Agent-based design:** Each feature is a stateless agent (module) with narrow input/output. The FastAPI backend orchestrates them; the React frontend is a static SPA served by nginx that never imports the pipeline directly.

**Service split (Docker):**

- **Backend** (`backend`) — FastAPI app exposing `/api/v1`. Owns the pipeline and the HTTP inference clients (Whisper, NER, diarization). CPU-only, built from `docker/Dockerfile.backend`.
- **Frontend** (`frontend`) — React SPA (node → nginx multi-stage build) in `frontend/`. Serves the compiled bundle and proxies `/api/v1` same-origin to the backend — no `BACKEND_HOST` env var needed. Built from `docker/Dockerfile.frontend`; `cd frontend && pnpm {dev,build,test,lint,typecheck}` for local development.

`nextext-cli` keeps the in-process path: it imports `nextext.pipeline` directly and runs end-to-end without needing a backend container. Lives in the backend image alongside the API.

**Playback.** The SPA has a slide-in media player (`frontend/src/components/player/MediaPanel.tsx`, mounted once in the Shell, driven by `lib/mediaPlayerStore.ts`). Clicking a timestamp in the Transcript or Visual context tab opens it and seeks there; the row under the playhead is highlighted and scrolled into view as playback advances (`hooks/useFollowActiveRow.ts`), so the highlight cannot drift off the fold. Following pauses the moment the reader scrolls by hand and resumes on the next timestamp click or seek — the store's `followSeq` counter is that signal. It is non-modal by design — no backdrop, no `aria-modal` — and the canvas reserves room for it on wide screens, so the page stays usable. `TranscriptSegment` carries `start_seconds`/`end_seconds` (numeric twins of the `str(timedelta)` display strings) so the client seeks without re-implementing the parser.

**Pipeline flow (server-side):**

1. **Transcription** (always-on) → every upload is re-encoded to 16 kHz mono FLAC (`nextext/core/audio.py`, PyAV) so libsndfile-only Whisper servers can decode it → external Whisper API (`/v1/audio/transcriptions`, always in the source language) behind an external `/vad` speech guard (defaults to the central endpoint; `VAD_API_BASE=off` skips it), + speaker diarization via the out-of-process `/diarize` HTTP service — on by default and auto-detecting the speaker count (no speaker bounds sent), bypassable per job (`diarize=false` / CLI `--no-diarize`); VAD-gating of the turns (cropping to the Silero speech timeline, dropping music/noise the diarizer over-detects as speech) happens server-side in the diarize backend (`DIARIZE_VAD_URL`, on by default in the full vllm-service stack), so turns arrive pre-gated; they are then aligned onto the transcript at the word level (falling back to segment-level overlap when the endpoint returns no word timestamps), with segments overlapping no turn inheriting the temporally nearest turn's speaker instead of rendering as `Unknown`, and finally renumbered to contiguous `Speaker N` by first appearance **in the assembled transcript**, so labels always read in order (Speaker 1, 2, 3 … top to bottom); the speaker column is omitted entirely when ≤1 distinct speaker is detected; low-punctuation transcripts (e.g. Arabic) are re-segmented into one sentence per row via `TEXT_MODEL` before merge/translate (`NEXTEXT_SENTENCE_RESTORE`, default on) → `pd.DataFrame`
2. **Translation** (optional) → LLM-based segment translation, directly source → target for any target language, via `InferencePipeline`. Whisper's audio-translate task is not used.
3. **Word-level analysis** (optional) → word counts, named entities via the out-of-process `/gliner` HTTP service, word clouds
4. **Summarization** (optional) → LLM summary via `InferencePipeline`. For a video file, the keyframes sampled in step 1 are first captioned one request at a time via `TEXT_MODEL`'s vision path (`call_vision`), and the timestamped captions are prepended to the transcript as a "Visual context" block, so the summary covers what was shown as well as what was said (`NEXTEXT_VISUAL_SUMMARY`, default on; needs a vision-capable `TEXT_MODEL`). No job option gates it — it applies whenever a summary was requested and the file actually has frames. Fail-soft: a text-only model aborts captioning after one request and the job falls back to an audio-only summary
5. **Hate-speech detection** (optional) → per-segment LLM classification
6. **Artifacts** → backend renders `.txt`, `.csv`, `.xlsx`, `.png`, `.jsonl`, ZIP on demand at `/api/v1/jobs/{id}/artifacts/{name}`

**HTTP API (`/api/v1`):**

- `POST /jobs` (multipart: `file` + JSON `options`) — queue a new job; returns `{job_id}`.
- `GET /jobs` — list the caller's in-memory jobs, newest first. The frontend calls this on load to re-discover and resume its jobs after a browser reload.
- `GET /jobs/{id}` — point-in-time snapshot (owner-scoped).
- `GET /jobs/{id}/events` — SSE stream of stage transitions (owner-scoped); replays event history on connect so a reattached client resumes mid-run.
- `GET /jobs/{id}/media?token=…` — the original upload, streamed for in-browser playback with HTTP Range (`206`) so the player can seek. Authorized by a per-job capability token, **not** by the principal: a `<video>`/`<audio>` element cannot attach the trusted header. The token is minted at job creation, handed out only on the owner-scoped snapshot (`JobResult.media_url`), kept off `GET /jobs`, and revoked by `DELETE`. Every failure is a `404` so a wrong token cannot probe which jobs exist.
- `GET /jobs/{id}/artifacts/{name}` — binary download (transcript.csv/xlsx/txt, translation.txt, summary.txt, visual_context.txt, wordcounts.csv/xlsx, entities.csv/xlsx, wordcloud.png, keyframes.zip, hate_speech.csv/xlsx, docint.jsonl, archive.zip). Owner-scoped.
- `DELETE /jobs/{id}` — cleanup (owner-scoped).
- `GET /health`, `GET /languages` — meta endpoints.
- `GET /metrics` — Prometheus exposition (aggregate request/latency counters plus the job-outcome counters below; no transcript or user data); unauthenticated, scraped by obs-plane over `inference-net`.

**Empty / not-processed outcomes.** A file with no processable speech is not a failure: the job **completes**, and `skipped: true` plus a typed `skip_reason_code` say why. The vocabulary lives in `nextext/core/outcomes.py`: `vad_no_speech` (the `/vad` guard rejected the file), `asr_empty_transcript` (Whisper returned nothing), `asr_all_segments_filtered` (every segment failed the `no_speech_prob` filter). Failures carry a typed `error_code` — `undecodable_media` or `internal` — while `error` stays the static `"Job failed."` so no detail leaks. Both codes appear on `GET /jobs/{id}`, on `GET /jobs` list items (the SPA's only source after a browser reload), and on the terminal `job_completed` / `job_failed` SSE frames; the frontend localizes them (`frontend/src/lib/outcomeMessages.ts`) and never renders the backend's own English prose. The worker logs one job-scoped warning per skipped or failed job and increments `nextext_jobs_total{outcome}`, `nextext_jobs_skipped_total{reason}`, `nextext_jobs_failed_total{code}` (`nextext/api/metrics.py`). A skipped job renders no artifacts at all — `archive.zip` 404s like its siblings rather than returning an empty ZIP. `nextext-cli` reports the same outcome as **exit code 3** (not `2`, which argparse uses for usage errors).

Identity is resolved per request by `resolve_principal`: the trusted header (`NEXTEXT_AUTH_HEADER`, default `X-Auth-User`) if present, else `NEXTEXT_DEFAULT_IDENTITY` (the dev / header-less fallback), else `401`. The value scopes the caller's in-memory jobs; cross-owner reads return `404` so existence never leaks. The React frontend mints a per-browser id and carries it in its URL (`?owner=<id>`) on first visit, reading it back on every reload so the identity survives browser refreshes. There is no authentication — the backend trusts whoever can reach `inference-net` — and no durable storage: jobs live only in memory.

**Key modules:**

- `nextext/api/main.py` — FastAPI factory, lifespan (boots the in-memory `JobManager`).
- `nextext/api/jobs.py` — `JobManager`, async workers bounded by a configurable `asyncio.Semaphore` (`NEXTEXT_JOB_CONCURRENCY`, default `1` = single in-flight job), SSE event broker. Holds all jobs in memory; `list_for_owner` powers the frontend's reload re-discovery.
- `nextext/api/identity.py` — `resolve_principal` FastAPI dependency. Reads the trusted header (`NEXTEXT_AUTH_HEADER`, default `X-Auth-User`); falls back to `NEXTEXT_DEFAULT_IDENTITY` for header-less/dev callers; returns `401` when neither is set. The React frontend carries the identity in its URL (`?owner=<id>`); there are no server-managed cookies. This is the single seam a real auth track would replace.
- `nextext/api/routes/` — `health`, `jobs` routers. Per-route ownership checks return `404` on cross-owner access so existence never leaks.
- `nextext/api/routes/jobs.py` — also serves `GET /jobs/{id}/media` via `FileResponse` (Starlette answers `Range` with `206`), with the MIME type guessed from the upload name.
- `nextext/api/artifacts.py` — Per-job artifact byte materializers (CSV/XLSX/PNG/JSONL/ZIP) rendered on demand from the in-memory `state.result`.
- `nextext/api/schemas.py` — Pydantic request/response models for jobs, snapshots, and the SSE event payloads.
- `nextext/api/metrics.py` — job-outcome Prometheus counters (`nextext_jobs_total`, `nextext_jobs_skipped_total`, `nextext_jobs_failed_total`), registered on the default registry the instrumentator exposes. Typed-code labels only.
- `nextext/core/outcomes.py` — the typed `SkipReason` / `FailureCode` vocabulary shared by the pipeline, API, and CLI, plus the English fallback prose for non-UI consumers.
- `nextext/cli.py` — CLI entry point (argparse), single-file processing in-process.
- `nextext/pipeline.py` — Shared pipeline functions connecting all agents. `transcription_pipeline` returns a `TranscriptionOutcome` (transcript, resolved source language, `skip_reason`); its `is_empty` property is the single definition of "no usable text" that the API worker and CLI both branch on.
- `nextext/core/transcription.py` — `ExternalWhisperTranscriber` (OpenAI-compatible audio API); the pre-upload speech guard is delegated to the external `/vad` service via `core/vad.py`. Records `skip_reason` when a run yields no segments, so the three causes stay distinguishable downstream.
- `nextext/core/audio.py` — audio-normalization agent: re-encodes any upload to 16 kHz mono FLAC via PyAV (bundled ffmpeg) before the Whisper call; fail-closed (`AudioDecodeError`) on undecodable input.
- `nextext/core/vad.py` — voice-activity-detection agent: fail-open HTTP client for the out-of-process `/vad` service — `has_speech` (pre-Whisper guard); an unset/unreachable endpoint transcribes everything.
- `nextext/core/diarization.py` — speaker-diarization agent: HTTP client for the out-of-process `/diarize` service; aligns turns onto the transcript at the word level (`build_speaker_segments`), falling back to segment-level overlap (`assign_speakers_by_overlap`) when word timestamps are unavailable, labels segments that overlap no turn with the temporally nearest turn's speaker (`fill_speakers_by_nearest_turn`, preferring the preceding turn on a tie, so no segment renders as `Unknown` while others carry speakers), then renumbers the finished transcript's speakers to contiguous `Speaker N` by first appearance in reading order (`renumber_speakers_by_appearance`; supersedes the earlier turn-order `canonicalize_speaker_labels` for what the transcript displays).
- `nextext/core/visual_context.py` — visual-context agent: captions sampled
  video keyframes via `InferencePipeline.call_vision` (`describe_keyframes`,
  one request per frame), downscales each frame first (`prepare_frame`), and
  renders the `[mm:ss] caption` block the summarizer receives
  (`format_visual_context`). Fail-soft throughout — a per-frame outage skips
  that caption, and a client rejection on the first frame (a text-only
  `TEXT_MODEL`) aborts the run after one round-trip.
- `nextext/core/keyframes.py` — keyframe sampler: `extract_keyframe_samples`
  returns timestamped `Keyframe` objects spanning the clip's full duration;
  `extract_keyframes` is the bytes-only wrapper the `keyframes.zip` artifact
  uses.
- `nextext/core/sentence_segmentation.py` — sentence-restoration agent: for
  low-punctuation transcripts (e.g. Arabic), re-segments the word stream into
  one segment per sentence via `TEXT_MODEL` (`restore_sentence_segments`), which
  returns `index:code` boundaries (never text) and appends the classified
  terminal mark (`.`/`؟`/`!`). Gated on `terminal_punctuation_ratio`; fail-soft.
- `nextext/core/ner.py` — named-entity-recognition agent: HTTP client for the out-of-process `/gliner` service (`extract_entities`).
- `nextext/core/translation.py` — LLM translation with prompt templates.
- `nextext/core/words.py` — NLP word-level analysis (spaCy word counts + word clouds).
- `nextext/core/hate_speech.py` — LLM-based hate-speech detection.
- `nextext/core/openai_cfg.py` — `InferencePipeline` for OpenAI-compatible LLM calls.
- `nextext/core/processing.py` — File I/O and export formatting (CLI).
- `nextext/utils/mappings/` — JSON config files for Whisper/spaCy model names, language codes.
- `nextext/utils/prompts/` — LLM prompt templates (system, translation, summary, hate_speech), localized per language under `en/` and `de/` (selected by `RESPONSE_LANGUAGE`, with `NEXTEXT_RESPONSE_LANGUAGE` honored as a deprecated fallback; English fallback).

## Environment

Key env vars (see `.env.example`):

- `OPENAI_API_KEY`, `OPENAI_API_BASE` — the **central** OpenAI-compatible endpoint; carries translation, summarization, and hate-speech detection (all on `TEXT_MODEL`), supplies the bearer token reused by the NER, diarization, and VAD clients, and is the fallback for every per-model endpoint below (Whisper verbatim; NER/diarization/VAD with one trailing `/v1` stripped, since they speak a plain service root).
- `WHISPER_API_BASE` / `WHISPER_API_KEY` / `WHISPER_MODEL` — dedicated Whisper endpoint (OpenAI SDK base incl. `/v1`); falls back to the central pair. Model defaults: `whisper-1` (openai), `openai/whisper-large-v3` (vllm). `INFERENCE_PROVIDER=ollama` has no transcription API, so it requires explicit `WHISPER_API_BASE` + `WHISPER_MODEL` (`load_whisper_env` raises otherwise).
- `NER_API_BASE` — root URL of the out-of-process `/gliner` NER service (e.g. `http://vllm-router:4000`); the client appends `/gliner`. Defaults to the central `OPENAI_API_BASE` (one trailing `/v1` stripped); set it only to point NER elsewhere. NER issues a request only when a job requests entities. The bearer token is reused from `OPENAI_API_KEY`. Fail-soft: errors degrade to empty entities. `NER_TIMEOUT` — per-request (per-chunk) timeout in seconds (default `120`).
- `DIARIZE_API_BASE` — root URL of the out-of-process `/diarize` service (e.g. `http://vllm-router:4000`); the client appends `/diarize`. Defaults to the central `OPENAI_API_BASE` (one trailing `/v1` stripped); set it only to override. Diarization runs by default for every job (auto-detecting the speaker count) unless the job sets `diarize=false`. Nextext requests word timestamps (`timestamp_granularities=["segment","word"]`) from the Whisper call and degrades gracefully to segment-level alignment when the ASR endpoint returns none. The bearer token is reused from `OPENAI_API_KEY`. Fail-soft: errors degrade to an unlabelled transcript. `DIARIZE_TIMEOUT` — per-request timeout in seconds (default `600`). See `nextext/core/diarization.py` for the `/diarize` request/response contract.
- `INFERENCE_PROVIDER` — `ollama` (default), `vllm`, or `openai`. Selects the Whisper model default and the Ollama `think` handling; prompts are provider-independent.
- `TEXT_MODEL` — LLM model name shared by translation, summarization, and hate-speech detection
- `RESPONSE_LANGUAGE` (backend + CLI + frontend) — Uniform federation UI-language switch: `en` (English, default) or `de` (German). Controls both the LLM output language (summaries, hate-speech rationales, prompts) and the SPA UI language. Backend selects the localized prompt subdirectory (`nextext/utils/prompts/<code>/`), frontend applies the `LanguageProvider` i18n translation. Missing locale files fall back to English; unrecognized values warn and fall back to `en`. Resolved by `load_language_env` in `nextext/utils/env_cfg.py` (backend + CLI) and the `LanguageProvider` (frontend).
- `NEXTEXT_RESPONSE_LANGUAGE` (backend + CLI, deprecated) — Legacy name for the output language setting. Kept as a fallback for one release; new deployments should use `RESPONSE_LANGUAGE` instead. If both are set, `RESPONSE_LANGUAGE` takes precedence.
- `SUMMARY_MAX_INPUT_TOKENS` (backend + CLI) — Max transcript tokens sent to `TEXT_MODEL` in a single summarize request. Longer transcripts are summarized map-reduce (chunk → summarize each → recursively summarize the combined partials) so no request overflows the chat model's context window; short transcripts take a single-shot path. Every request also caps output at 1024 tokens (`SUMMARY_MAX_OUTPUT_TOKENS` in `nextext/pipeline.py`). The token budget is converted to a character budget with a conservative ratio (`_CHARS_PER_TOKEN`), so lower it for token-dense scripts (e.g. CJK) or small `max_model_len` backends and raise it to reduce chunking. If a request still overflows, the budget auto-halves and retries (up to 4×), then fail-soft degrades to an empty summary rather than crashing the job. Invalid/≤0 values warn and fall back. Defaults to `6000`.
- `OLLAMA_THINK` — tri-state default for the Ollama `think` request field forwarded by `InferencePipeline.call_model` via `extra_body`. Accepts `1`/`true`/`yes`/`on` (enable), `0`/`false`/`no`/`off` (disable), or unset (omit field, model default). Honoured by Ollama-hosted reasoning models such as Qwen3; a no-op for `vllm`/`openai` providers. Per-call `think=` overrides the env default.
- `VAD_API_BASE` — root URL of the out-of-process `/vad` speech-guard service (e.g. `http://vllm-router:4000`); the client appends `/vad`. Defaults to the central `OPENAI_API_BASE` (one trailing `/v1` stripped), so the guard runs ahead of every transcription; set `VAD_API_BASE=off` (or `false`/`no`/`0`) to switch it off, or a URL to override. The bearer token is reused from `OPENAI_API_KEY`. Fail-open: an unreachable service degrades to transcribing anyway. `VAD_TIMEOUT` — per-request timeout in seconds (default `60`). See `nextext/core/vad.py` for the `/vad` request/response contract.
- `NEXTEXT_SENTENCE_RESTORE` / `SENTENCE_RESTORE_MIN_PUNCT_RATIO` (backend + CLI) —
  Sentence restoration for punctuation-poor transcripts. When on (default) and a
  transcript's terminal-punctuation density (marks ÷ words) is below
  `SENTENCE_RESTORE_MIN_PUNCT_RATIO` (default `0.01`), each contiguous speaker
  run is re-segmented into whole sentences by `TEXT_MODEL`, so rows are one
  sentence each (granular and a coherent translation unit) instead of a
  whole-speaker-turn blob. The model returns `index:code` boundaries — never
  text — so words/timestamps stay untouched; questions get `؟`, exclamations
  `!`, else `.`. Fail-soft: a model outage degrades to today's behavior. Resolved
  by `load_sentence_restore_env`. Set `NEXTEXT_SENTENCE_RESTORE=off` to disable.
- `NEXTEXT_VISUAL_SUMMARY` / `VISUAL_SUMMARY_MAX_FRAMES` / `VISUAL_SUMMARY_IMAGE_MAX_SIDE` (backend + CLI) —
  Visual context for video summaries. When on (default) and a summary is
  requested for a file with a video stream, the sampled keyframes are captioned
  by `TEXT_MODEL` and folded into the summary; captions also surface as
  `JobResult.frame_captions` and the `visual_context.txt` artifact. Requires a
  vision-capable `TEXT_MODEL` — a text-only one degrades to today's audio-only
  summary with one warning. `VISUAL_SUMMARY_MAX_FRAMES` (default `12`, clamped
  to `200`) bounds the per-job cost at one inference request per frame;
  `VISUAL_SUMMARY_IMAGE_MAX_SIDE` (default `1024`) bounds each frame's upload
  size. Resolved by `load_visual_summary_env`. Set `NEXTEXT_VISUAL_SUMMARY=off`
  to disable; `nextext-cli` also takes `--no-visual-context`.
- `NEXTEXT_OFFLINE=1` (default) — gates the spaCy/NLTK downloads (`is_offline()`); the only local downloads left. Offline + uncached spaCy model raises an actionable error.
- `NEXTEXT_HOST_PORT` (frontend, dev/override only) — host port published by `make up-dev` for the nginx frontend container. Defaults to `8501`; maps to nginx port `8080` (the unprivileged nginx image listens there — see Container hardening below).
- `NEXTEXT_CLIENT_MAX_BODY_SIZE` (frontend) — nginx `client_max_body_size` for the `/api/v1` upload proxy. Defaults to `8192m`.
- `NEXTEXT_API_HOST` / `NEXTEXT_API_PORT` (backend only) — uvicorn bind address. Defaults to `0.0.0.0:8000`.
- `NEXTEXT_DEFAULT_TARGET_LANG` (backend only) — Initial translation target language code surfaced by `GET /languages` as `default_target` and used to seed the frontend's "Target language" dropdown on a fresh browser. Must be a supported target code; an unsupported (or unset) value falls back to English (`en`). The frontend persists the user's own selection per-browser (localStorage), so it survives reloads and takes precedence over this default. Defaults to `en`.
- `NEXTEXT_MAX_UPLOAD_MB` (backend only) — Hard cap on per-file upload bytes. The backend streams the upload to disk in 1 MiB chunks (`_stream_upload_to_disk` in `nextext/api/routes/jobs.py`) and returns `413` once the cap is exceeded; unparseable values fall back to the default and `<1` clamps to `1`. The React frontend mirrors the *default* as an advisory client-side check only (`DEFAULT_MAX_FILE_MB` in `frontend/src/lib/uploadGuard.ts`) — it does not read the env var, and the backend is the enforcement point. The separate nginx body limit is `NEXTEXT_CLIENT_MAX_BODY_SIZE`. Defaults to `8192`.
- `NEXTEXT_JOB_CONCURRENCY` (backend only) — Max jobs the in-memory `JobManager` runs concurrently (`asyncio.Semaphore`). Defaults to `1` (serial, one in-flight job — the historical behavior); raise it to overlap jobs, bounded by container CPU (PyAV decode per job) and the external inference services' capacity. Unparseable/`<1` values clamp to `1`. Resolved by `load_job_concurrency` in `nextext/utils/env_cfg.py`.
- `KEYFRAMES_PER_MINUTE` (backend only) — Default keyframes sampled per minute of video, applied to `JobOptions.keyframes_per_minute` only when a job-creation request omits the field (an explicit per-request value always wins). Invalid values warn and fall back; negatives clamp to `0`. Resolved by `load_keyframe_defaults` in `nextext/utils/env_cfg.py`. Defaults to `4`.
- `KEYFRAMES_MAX` (backend only) — Default hard ceiling on keyframes returned per clip, applied to `JobOptions.keyframes_max` only when a request omits it. Clamped to `[0, 200]` (the schema's hard cap; larger values warn and clamp to `200`); an explicit per-request value still overrides. Defaults to `20`.
- There is no combined-batch size cap. The old `NEXTEXT_MAX_BATCH_MB` existed for Streamlit's `file_uploader`, which held a whole multi-file selection in the frontend process's memory; the React SPA streams each file through nginx instead, so only the **per-file** guard above applies. Large local batches still belong in `nextext-cli`, which reads from disk and never buffers whole files.
- `NEXTEXT_AUTH_HEADER` (backend + frontend) — Name of the trusted identity header. Defaults to `X-Auth-User`. Both sides read the same variable so they agree on the header.
- `NEXTEXT_DEFAULT_IDENTITY` (backend only) — Fallback identity for header-less / developer callers. Unset by default, so a request without the trusted header gets `401`.

## Docker

Docker assets live under `docker/`. `docker/compose.yaml` defines two services — no profiles, no GPU reservations:

- `backend` — built from `docker/Dockerfile.backend`, multi-stage `uv` build (no extras; runtime apt is `curl` only — all inference, including the VAD guard, is external; audio normalization uses the PyAV wheel, so no apt audio tooling is added). Runs `uvicorn nextext.api.main:app` with a `HEALTHCHECK` against `/api/v1/health`. Reachable only on the `nextext-net` network by default; no host port is published.
- `frontend` — React SPA compiled and served by nginx. Built from `docker/Dockerfile.frontend` (node build → nginx image). The nginx config proxies `/api/v1` same-origin to the backend, so browser uploads stream through nginx without buffering whole files in any Python process. The base `docker/compose.yaml` is the production shape and publishes no host ports; `docker/compose.override.yaml` (layered by `make up-dev`) publishes nginx on `${NEXTEXT_HOST_PORT:-8501}`.

The stack shares `inference-net` with the inference provider (vllm-service / Ollama). The `Makefile` is the entry point — it points Compose at `docker/compose.yaml`, since a bare `docker compose` from the repo root no longer finds it. Run `make volumes` (one-time, creates the external `nltk-cache`/`spacy-cache` volumes), then `make build && make up` for production shape, or `make build && make up-dev` (or just `make dev`) to publish the frontend on the host. `make up`/`make up-dev` are detached and never build (`--no-build`), so build the images first (in prod, load or pull them). `make bundle` writes an image tarball built from the latest annotated release tag (production); `make bundle-dev` bundles the current working tree instead (dev/soak).

The React SPA source lives in `frontend/`; run `cd frontend && pnpm {dev,build,test,lint,typecheck}` for local development without Docker.

**Container hardening (deploy ADR 0001):** both containers run non-root with
read-only root filesystems — the backend as uid `10001` (`app`,
`HOME=/home/app`; `NLTK_DATA`/`SPACY_MODEL_DIR` point at the cache volumes
under `/home/app`, `/tmp` is a disk-backed scratch volume sized for multi-GB
job media, and `MPLCONFIGDIR=/tmp/matplotlib` keeps matplotlib's import-time
config/font cache off the read-only `$HOME/.config`), the frontend on
`nginxinc/nginx-unprivileged` as uid `101`
listening on **:8080** (the edge gateway's `nextext-frontend` upstream must
match). Compose applies `no-new-privileges` + `cap_drop: ALL` via the
`x-hardened` anchor. On existing hosts the `nltk-cache`/`spacy-cache` volumes
need a one-time `chown -R 10001:10001` (runbook in the `deploy` repo).

## Persistence model

Jobs live only in memory. `JobManager` holds them in a dict keyed by `job_id` and scoped by `owner_id`; there is no SQLite index, no on-disk artifacts, and no TTL sweeper. A job is retained until the owner `DELETE`s it or the backend process exits — nothing ever cuts off a long-running job.

Reload resilience comes from the identity, not from storage. The owner id survives a browser refresh in the page URL (`?owner=<id>`), so on load the frontend calls `GET /jobs` to re-discover the caller's jobs and resumes them: it re-subscribes to any still running (the SSE broker replays each job's event history on connect) and re-renders those already finished. A run therefore survives a browser reload during processing, but not a backend restart.

The upload itself outlives the pipeline: it stays at `state.file_path` until the owner `DELETE`s the job or the backend exits, which is what makes playback possible after a job completes. Nothing else reclaims it.

Artifacts (`.csv`/`.xlsx`/`.png`/`.jsonl`/`.zip`) are materialised on demand from the in-memory `state.result` by `nextext/api/artifacts.py`; they are never written to disk.

## Deployment: edge-plane gateway sub-path

The Nextext SPA is served in production under the canonical `/nextext/`
sub-path behind the `edge-plane` gateway, not at its own vhost root. The
`frontend` service joins the external `edge-net` network (alongside its
existing `nextext-net` membership) as alias `nextext-frontend`, which is how
the gateway reaches it. Vite is built with `base: '/nextext/'`, `API_BASE`
derives from `BASE_URL` (`frontend/src/api/client.ts`'s `apiBase()`), the
`BrowserRouter` uses a matching `basename`, and the frontend's nginx config
strips the `/nextext` prefix internally before falling through to the
existing root-anchored locations (the SSE job-events endpoint included),
redirecting bare `/` to `/nextext/`. The gateway is the sole production entry
point and is what injects `X-Auth-User` for the backend's trusted-header
principal seam (`nextext/api/identity.py`) — production leaves
`NEXTEXT_DEFAULT_IDENTITY` unset so requests without that header are rejected
as unauthenticated; any dev-only fallback stays dev-only.

## Commits

- Prefer multiple small topical commits over a single catch-all commit.
- Each commit message should describe a single logical change (refactor, fix, feat, docs, test).
