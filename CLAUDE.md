# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## Docstrings & Style

- All new/modified Python functions must have Google-style docstrings.
- Python 3.12 is the target; prefer explicit types and distinct variable names across branches to satisfy pyrefly.

## Architecture

**Agent-based design:** Each feature is a stateless agent (module) with narrow input/output. The FastAPI backend orchestrates them; the React frontend is a static SPA served by nginx that never imports the pipeline directly.

**Service split (Docker):**

- **Backend** (`backend`) — FastAPI app exposing `/api/v1`. Owns the pipeline and the HTTP inference clients (Whisper, NER, diarization). CPU-only, built from `docker/Dockerfile.backend`.
- **Frontend** (`frontend`) — React SPA (node → nginx multi-stage build) in `frontend/`. Serves the compiled bundle and proxies `/api/v1` same-origin to the backend — no `BACKEND_HOST` env var needed. Built from `docker/Dockerfile.frontend`; `cd frontend && pnpm {dev,build,test,lint,typecheck}` for local development.

`nextext-cli` keeps the in-process path: it imports `nextext.pipeline` directly and runs end-to-end without needing a backend container. Lives in the backend image alongside the API.

**Pipeline flow (server-side):**

1. **Transcription** (always-on) → every upload is re-encoded to 16 kHz mono FLAC (`nextext/core/audio.py`, PyAV) so libsndfile-only Whisper servers can decode it → external Whisper API (`/v1/audio/transcriptions`, always in the source language) behind an external `/vad` speech guard (defaults to the central endpoint; `VAD_API_BASE=off` skips it), + optional speaker diarization via the out-of-process `/diarize` HTTP service (when `max speakers > 1`) → `pd.DataFrame`
2. **Translation** (optional) → LLM-based segment translation, directly source → target for any target language, via `InferencePipeline`. Whisper's audio-translate task is not used.
3. **Word-level analysis** (optional) → word counts, named entities via the out-of-process `/gliner` HTTP service, word clouds
4. **Summarization** (optional) → LLM summary via `InferencePipeline`
5. **Hate-speech detection** (optional) → per-segment LLM classification
6. **Artifacts** → backend renders `.txt`, `.csv`, `.xlsx`, `.png`, `.jsonl`, ZIP on demand at `/api/v1/jobs/{id}/artifacts/{name}`

**HTTP API (`/api/v1`):**

- `POST /jobs` (multipart: `file` + JSON `options`) — queue a new job; returns `{job_id}`.
- `GET /jobs` — list the caller's in-memory jobs, newest first. The frontend calls this on load to re-discover and resume its jobs after a browser reload.
- `GET /jobs/{id}` — point-in-time snapshot (owner-scoped).
- `GET /jobs/{id}/events` — SSE stream of stage transitions (owner-scoped); replays event history on connect so a reattached client resumes mid-run.
- `GET /jobs/{id}/artifacts/{name}` — binary download (transcript.csv/xlsx, summary.txt, wordcounts.csv/xlsx, entities.csv/xlsx, wordcloud.png, hate_speech.csv/xlsx, docint.jsonl, archive.zip). Owner-scoped.
- `DELETE /jobs/{id}` — cleanup (owner-scoped).
- `GET /health`, `GET /languages` — meta endpoints.

Identity is resolved per request by `resolve_principal`: the trusted header (`NEXTEXT_AUTH_HEADER`, default `X-Auth-User`) if present, else `NEXTEXT_DEFAULT_IDENTITY` (the dev / header-less fallback), else `401`. The value scopes the caller's in-memory jobs; cross-owner reads return `404` so existence never leaks. The React frontend mints a per-browser id and carries it in its URL (`?owner=<id>`) on first visit, reading it back on every reload so the identity survives browser refreshes. There is no authentication — the backend trusts whoever can reach `inference-net` — and no durable storage: jobs live only in memory.

**Key modules:**

- `nextext/api/main.py` — FastAPI factory, lifespan (boots the in-memory `JobManager`).
- `nextext/api/jobs.py` — `JobManager`, async worker (single in-flight job via `asyncio.Semaphore(1)`), SSE event broker. Holds all jobs in memory; `list_for_owner` powers the frontend's reload re-discovery.
- `nextext/api/identity.py` — `resolve_principal` FastAPI dependency. Reads the trusted header (`NEXTEXT_AUTH_HEADER`, default `X-Auth-User`); falls back to `NEXTEXT_DEFAULT_IDENTITY` for header-less/dev callers; returns `401` when neither is set. The React frontend carries the identity in its URL (`?owner=<id>`); there are no server-managed cookies. This is the single seam a real auth track would replace.
- `nextext/api/routes/` — `health`, `jobs` routers. Per-route ownership checks return `404` on cross-owner access so existence never leaks.
- `nextext/api/artifacts.py` — Per-job artifact byte materializers (CSV/XLSX/PNG/JSONL/ZIP) rendered on demand from the in-memory `state.result`.
- `nextext/api/schemas.py` — Pydantic request/response models for jobs, snapshots, and the SSE event payloads.
- `nextext/cli.py` — CLI entry point (argparse), single-file processing in-process.
- `nextext/pipeline.py` — Shared pipeline functions connecting all agents.
- `nextext/core/transcription.py` — `ExternalWhisperTranscriber` (OpenAI-compatible audio API); the pre-upload speech guard is delegated to the external `/vad` service via `core/vad.py`.
- `nextext/core/audio.py` — audio-normalization agent: re-encodes any upload to 16 kHz mono FLAC via PyAV (bundled ffmpeg) before the Whisper call; fail-closed (`AudioDecodeError`) on undecodable input.
- `nextext/core/vad.py` — voice-activity-detection agent: fail-open HTTP client for the out-of-process `/vad` service (`has_speech`); an unset/unreachable endpoint transcribes everything.
- `nextext/core/diarization.py` — speaker-diarization agent: HTTP client for the out-of-process `/diarize` service + client-side speaker/segment overlap alignment (`assign_speakers_by_overlap`).
- `nextext/core/ner.py` — named-entity-recognition agent: HTTP client for the out-of-process `/gliner` service (`extract_entities`).
- `nextext/core/translation.py` — LLM translation with prompt templates.
- `nextext/core/words.py` — NLP word-level analysis (spaCy word counts + word clouds).
- `nextext/core/hate_speech.py` — LLM-based hate-speech detection.
- `nextext/core/openai_cfg.py` — `InferencePipeline` for OpenAI-compatible LLM calls.
- `nextext/core/processing.py` — File I/O and export formatting (CLI).
- `nextext/utils/mappings/` — JSON config files for Whisper/spaCy model names, language codes.
- `nextext/utils/prompts/` — LLM prompt templates (system, translation, summary, hate_speech), localized per language under `en/` and `de/` (selected by `NEXTEXT_RESPONSE_LANGUAGE`, English fallback).

See `AGENTS.md` for detailed agent documentation including I/O contracts and how to add new agents.

## Environment

Key env vars (see `.env.example`):

- `OPENAI_API_KEY`, `OPENAI_API_BASE` — the **central** OpenAI-compatible endpoint; carries translation, summarization, and hate-speech detection (all on `TEXT_MODEL`), supplies the bearer token reused by the NER, diarization, and VAD clients, and is the fallback for every per-model endpoint below (Whisper verbatim; NER/diarization/VAD with one trailing `/v1` stripped, since they speak a plain service root).
- `WHISPER_API_BASE` / `WHISPER_API_KEY` / `WHISPER_MODEL` — dedicated Whisper endpoint (OpenAI SDK base incl. `/v1`); falls back to the central pair. Model defaults: `whisper-1` (openai), `openai/whisper-large-v3` (vllm). `INFERENCE_PROVIDER=ollama` has no transcription API, so it requires explicit `WHISPER_API_BASE` + `WHISPER_MODEL` (`load_whisper_env` raises otherwise).
- `NER_API_BASE` — root URL of the out-of-process `/gliner` NER service (e.g. `http://vllm-router:4000`); the client appends `/gliner`. Defaults to the central `OPENAI_API_BASE` (one trailing `/v1` stripped); set it only to point NER elsewhere. NER issues a request only when a job requests entities. The bearer token is reused from `OPENAI_API_KEY`. Fail-soft: errors degrade to empty entities. `NER_TIMEOUT` — per-request (per-chunk) timeout in seconds (default `120`).
- `DIARIZE_API_BASE` — root URL of the out-of-process `/diarize` service (e.g. `http://vllm-router:9000`); the client appends `/diarize`. Defaults to the central `OPENAI_API_BASE` (one trailing `/v1` stripped); set it only to override. Diarization runs only when `max speakers > 1`. The bearer token is reused from `OPENAI_API_KEY`. Fail-soft: errors degrade to an unlabelled transcript. `DIARIZE_TIMEOUT` — per-request timeout in seconds (default `600`). See `nextext/core/diarization.py` for the `/diarize` request/response contract.
- `INFERENCE_PROVIDER` — `ollama` (default), `vllm`, or `openai`. Selects the Whisper model default and the Ollama `think` handling; prompts are provider-independent.
- `TEXT_MODEL` — LLM model name shared by translation, summarization, and hate-speech detection
- `NEXTEXT_RESPONSE_LANGUAGE` (backend + CLI) — Output language for summaries and hate-speech rationales: `en` (default) or `de`. Selects the localized prompt subdirectory (`nextext/utils/prompts/<code>/`) supplying the system, summary, and hate-speech prompts, so summaries follow this operator-chosen language rather than the transcription source or the translation target. Translation output is unaffected — it follows the per-job target language. Missing locale files fall back to the English copy under `en/`; unrecognised values warn and fall back to `en`. Resolved by `load_language_env` in `nextext/utils/env_cfg.py`.
- `SUMMARY_MAX_INPUT_TOKENS` (backend + CLI) — Max transcript tokens sent to `TEXT_MODEL` in a single summarize request. Longer transcripts are summarized map-reduce (chunk → summarize each → recursively summarize the combined partials) so no request overflows the chat model's context window; short transcripts take a single-shot path. Every request also caps output at 1024 tokens (`SUMMARY_MAX_OUTPUT_TOKENS` in `nextext/pipeline.py`). The token budget is converted to a character budget with a conservative ratio (`_CHARS_PER_TOKEN`), so lower it for token-dense scripts (e.g. CJK) or small `max_model_len` backends and raise it to reduce chunking. If a request still overflows, the budget auto-halves and retries (up to 4×), then fail-soft degrades to an empty summary rather than crashing the job. Invalid/≤0 values warn and fall back. Defaults to `6000`.
- `OLLAMA_THINK` — tri-state default for the Ollama `think` request field forwarded by `InferencePipeline.call_model` via `extra_body`. Accepts `1`/`true`/`yes`/`on` (enable), `0`/`false`/`no`/`off` (disable), or unset (omit field, model default). Honoured by Ollama-hosted reasoning models such as Qwen3; a no-op for `vllm`/`openai` providers. Per-call `think=` overrides the env default.
- `VAD_API_BASE` — root URL of the out-of-process `/vad` speech-guard service (e.g. `http://vllm-router:7000`); the client appends `/vad`. Defaults to the central `OPENAI_API_BASE` (one trailing `/v1` stripped), so the guard runs ahead of every transcription; set `VAD_API_BASE=off` (or `false`/`no`/`0`) to switch it off, or a URL to override. The bearer token is reused from `OPENAI_API_KEY`. Fail-open: an unreachable service degrades to transcribing anyway. `VAD_TIMEOUT` — per-request timeout in seconds (default `60`). See `nextext/core/vad.py` for the `/vad` request/response contract.
- `NEXTEXT_OFFLINE=1` (default) — gates the spaCy/NLTK downloads (`is_offline()`); the only local downloads left. Offline + uncached spaCy model raises an actionable error.
- `NEXTEXT_HOST_PORT` (frontend, dev/override only) — host port published by `make up-dev` for the nginx frontend container. Defaults to `8501`; maps to nginx port 80.
- `NEXTEXT_CLIENT_MAX_BODY_SIZE` (frontend) — nginx `client_max_body_size` for the `/api/v1` upload proxy. Defaults to `8192m`.
- `NEXTEXT_API_HOST` / `NEXTEXT_API_PORT` (backend only) — uvicorn bind address. Defaults to `0.0.0.0:8000`.
- `NEXTEXT_DEFAULT_TARGET_LANG` (backend only) — Initial translation target language code surfaced by `GET /languages` as `default_target` and used to seed the frontend's "Target language" dropdown on a fresh browser. Must be a supported target code; an unsupported (or unset) value falls back to English (`en`). The frontend persists the user's own selection per-browser (localStorage), so it survives reloads and takes precedence over this default. Defaults to `en`.
- `NEXTEXT_MAX_UPLOAD_MB` (backend only) — Hard cap on per-file upload bytes (backend streams the upload to disk in 1 MiB chunks; also drives Streamlit's `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` per-file cap in compose). Defaults to `8192`.
- `KEYFRAMES_PER_MINUTE` (backend only) — Default keyframes sampled per minute of video, applied to `JobOptions.keyframes_per_minute` only when a job-creation request omits the field (an explicit per-request value always wins). Invalid values warn and fall back; negatives clamp to `0`. Resolved by `load_keyframe_defaults` in `nextext/utils/env_cfg.py`. Defaults to `4`.
- `KEYFRAMES_MAX` (backend only) — Default hard ceiling on keyframes returned per clip, applied to `JobOptions.keyframes_max` only when a request omits it. Clamped to `[0, 200]` (the schema's hard cap; larger values warn and clamp to `200`); an explicit per-request value still overrides. Defaults to `20`.
- `NEXTEXT_MAX_BATCH_MB` (frontend only) — Cap on the combined size of one multi-file upload selection. Streamlit's `file_uploader` holds the whole batch in the frontend process's memory at once, so the UI refuses an oversized batch (with an actionable message pointing at `nextext-cli`) instead of OOM-crashing. Defaults to `2048`. Large local batches belong in `nextext-cli`, which reads from disk and never buffers whole files.
- `NEXTEXT_AUTH_HEADER` (backend + frontend) — Name of the trusted identity header. Defaults to `X-Auth-User`. Both sides read the same variable so they agree on the header.
- `NEXTEXT_DEFAULT_IDENTITY` (backend only) — Fallback identity for header-less / developer callers. Unset by default, so a request without the trusted header gets `401`.

## Docker

Docker assets live under `docker/`. `docker/compose.yaml` defines two services — no profiles, no GPU reservations:

- `backend` — built from `docker/Dockerfile.backend`, multi-stage `uv` build (no extras; runtime apt is `curl` only — all inference, including the VAD guard, is external; audio normalization uses the PyAV wheel, so no apt audio tooling is added). Runs `uvicorn nextext.api.main:app` with a `HEALTHCHECK` against `/api/v1/health`. Reachable only on the `nextext-net` network by default; no host port is published.
- `frontend` — React SPA compiled and served by nginx. Built from `docker/Dockerfile.frontend` (node build → nginx image). The nginx config proxies `/api/v1` same-origin to the backend, so browser uploads stream through nginx without buffering whole files in any Python process. The base `docker/compose.yaml` is the production shape and publishes no host ports; `docker/compose.override.yaml` (layered by `make up-dev`) publishes nginx on `${NEXTEXT_HOST_PORT:-8501}`.

The stack shares `inference-net` with the inference provider (vllm-service / Ollama). The `Makefile` is the entry point — it points Compose at `docker/compose.yaml`, since a bare `docker compose` from the repo root no longer finds it. Run `make volumes` (one-time, creates the external `nltk-cache`/`spacy-cache` volumes), then `make build && make up` for production shape, or `make build && make up-dev` (or just `make dev`) to publish the frontend on the host. `make up`/`make up-dev` are detached and never build (`--no-build`), so build the images first (in prod, load or pull them).

The React SPA source lives in `frontend/`; run `cd frontend && pnpm {dev,build,test,lint,typecheck}` for local development without Docker.

## Persistence model

Jobs live only in memory. `JobManager` holds them in a dict keyed by `job_id` and scoped by `owner_id`; there is no SQLite index, no on-disk artifacts, and no TTL sweeper. A job is retained until the owner `DELETE`s it or the backend process exits — nothing ever cuts off a long-running job.

Reload resilience comes from the identity, not from storage. The owner id survives a browser refresh in the page URL (`?owner=<id>`), so on load the frontend calls `GET /jobs` to re-discover the caller's jobs and resumes them: it re-subscribes to any still running (the SSE broker replays each job's event history on connect) and re-renders those already finished. A run therefore survives a browser reload during processing, but not a backend restart.

Artifacts (`.csv`/`.xlsx`/`.png`/`.jsonl`/`.zip`) are materialised on demand from the in-memory `state.result` by `nextext/api/artifacts.py`; they are never written to disk.

## Commits

- Prefer multiple small topical commits over a single catch-all commit.
- Each commit message should describe a single logical change (refactor, fix, feat, docs, test).
