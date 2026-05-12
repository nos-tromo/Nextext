# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nextext is a modular audio analysis toolkit that transcribes, translates, and analyzes natural language from audio/video files. It uses openai-whisper for transcription, pyannote-audio for diarization, GLiNER for named-entity recognition, spaCy/NLTK for word-level NLP, and LLMs (Ollama, vLLM, or OpenAI-compatible endpoints) for translation, summarization, and hate-speech detection.

## Project Context

- WhisperX has been removed from this project; use openai-whisper + pyannote.
- Target torch install is split by extras (cpu/cuda) via conflicts in pyproject.toml.
- Docker base image is pinned to `python:3.12.10-slim-bookworm` across all Dockerfiles.

## Commands

```bash
# Install dependencies
uv sync                    # production deps
uv sync --group dev        # include dev deps (pytest, ruff, mypy, pre-commit)
uv sync --group frontend   # frontend-only deps (no torch / openai-whisper / spaCy)

# Run the app
uv run nextext             # Streamlit web UI on port 8501 (talks to backend over HTTP)
uv run nextext-api         # FastAPI backend on port 8000
uv run nextext-cli -f <file> [args]  # CLI mode (in-process, no backend required)

# Preload ML models
uv run load-models

# Tests
uv run pytest              # run full test suite
uv run pytest tests/test_pipeline.py  # single test file
uv run pytest -k "test_name"          # single test by name

# Linting & formatting (also enforced by pre-commit hooks)
ruff check --fix           # lint with auto-fix
ruff format                # format code
mypy --no-incremental --ignore-missing-imports --disable-error-code=import-untyped --disable-error-code=attr-defined --disable-error-code=assignment nextext/
```

## Testing

- Always run the full test suite (`pytest`) after making changes and report pass/fail counts.
- When tests fail, fix the root cause rather than patching tests to match stale/removed code.
- Verify with `pre-commit run --all-files` (mypy, lint, docstrings) before declaring work complete.

Tests are in `tests/` using pytest with monkeypatch fixtures for mocking ML models and inference. Tests simulate Docker detection and environment configuration. No GPU or model downloads required for tests.

## Docstrings & Style

- All new/modified Python functions must have Google-style docstrings.
- Python 3.12 is the target; prefer explicit types and distinct variable names across branches to satisfy mypy.

## Architecture

**Agent-based design:** Each feature is a stateless agent (module) with narrow input/output. The FastAPI backend orchestrates them; the Streamlit frontend is a thin HTTP client that never imports the pipeline directly.

**Service split (Docker):**

- **Backend** (`backend-cpu` / `backend-cuda`) — FastAPI app exposing `/api/v1`. Owns the pipeline, GPU model registry, and model caches. Built from `Dockerfile.backend.{cpu,cuda}`.
- **Frontend** (`frontend-cpu` / `frontend-cuda`) — Streamlit-only container. Talks to the backend via `BACKEND_HOST` (default `http://backend:8000`). Built from `Dockerfile.frontend` (single-stage, ships only the `frontend` dependency group; no torch, no ML libs).

`nextext-cli` keeps the in-process path: it imports `nextext.pipeline` directly and runs end-to-end without needing a backend container. Lives in the backend image alongside the API.

**Pipeline flow (server-side):**

1. **Transcription** (always-on) → openai-whisper transcription + optional pyannote diarization → `pd.DataFrame`
2. **Translation** (optional) → LLM-based segment translation via `InferencePipeline`
3. **Word-level analysis** (optional) → word counts, GLiNER named entities, word clouds
4. **Summarization** (optional) → LLM summary via `InferencePipeline`
5. **Hate-speech detection** (optional) → per-segment LLM classification
6. **Artifacts** → backend renders `.txt`, `.csv`, `.xlsx`, `.png`, `.jsonl`, ZIP on demand at `/api/v1/jobs/{id}/artifacts/{name}`

**HTTP API (`/api/v1`):**

- `POST /jobs` (multipart: `file` + JSON `options`) — queue a new job; returns `{job_id}`. `options.persist=true` opts in to durable storage; ephemeral by default.
- `GET /jobs` — list the caller's persistent jobs, newest first.
- `GET /jobs/{id}` — point-in-time snapshot (owner-scoped).
- `GET /jobs/{id}/events` — SSE stream of stage transitions (owner-scoped).
- `GET /jobs/{id}/artifacts/{name}` — binary download (transcript.csv/xlsx, summary.txt, wordcounts.csv/xlsx, entities.csv/xlsx, wordcloud.png, hate_speech.csv/xlsx, docint.jsonl, archive.zip). Owner-scoped.
- `DELETE /jobs/{id}` — cleanup (owner-scoped).
- `GET /health`, `GET /languages` — meta endpoints.

Every request carries an `X-Owner-Id` header (UUID4 hex) the browser stores in `localStorage`. The header is the only thing the backend uses to scope rows; clearing site data produces a fresh identity, and any persistent rows owned by the previous identity become unreachable to the new one. The identity survives tab close and browser restart, since `localStorage` outlives the page. There is still no authentication — the backend trusts whoever can reach `inference-net`.

**Key modules:**

- `nextext/api/main.py` — FastAPI factory, lifespan (boots `JobManager` + persistence repository).
- `nextext/api/jobs.py` — `JobManager`, async worker (single in-flight job via `asyncio.Semaphore(1)`), SSE event broker. Owns the bridge to durable storage when a job opts in.
- `nextext/api/identity.py` — Header-based `get_owner_id` FastAPI dependency. Reads the `X-Owner-Id` request header (UUID4 hex), rejects missing/malformed values with 400. The browser owns the identity (kept in `localStorage` by the Streamlit frontend); there are no server-managed cookies.
- `nextext/api/persistence.py` — `JobRepository` protocol + `SqliteJobRepository` implementation (WAL mode), `ArtifactStore` filesystem helper. Postgres-ready by design — replace the implementation, keep the protocol.
- `nextext/api/routes/` — `health`, `jobs` routers. Per-route ownership checks return `404` on cross-owner access so existence never leaks.
- `nextext/api/artifacts.py` — Per-job artifact byte materializers (CSV/XLSX/PNG/JSONL/ZIP). Lazily hydrates from disk for persistent jobs rehydrated at startup.
- `nextext/api/schemas.py` — Pydantic request/response models. `JobOptions.persist` toggles durable storage per submission.
- `nextext/frontend/app.py` — Streamlit entry point talking to the backend.
- `nextext/frontend/client.py` — `BackendClient` (httpx wrapper) with SSE parsing.
- `nextext/frontend/state.py` — Pure UI helpers (no pipeline imports).
- `nextext/app.py` — Compatibility shim re-exporting helpers from the locations above; preserves the historical import surface for tests and external callers.
- `nextext/cli.py` — CLI entry point (argparse), single-file processing in-process.
- `nextext/pipeline.py` — Shared pipeline functions connecting all agents.
- `nextext/core/transcription.py` — openai-whisper transcription & pyannote diarization.
- `nextext/core/translation.py` — LLM translation with prompt templates.
- `nextext/core/words.py` — NLP word-level analysis (spaCy + GLiNER NER).
- `nextext/core/hate_speech.py` — LLM-based hate-speech detection.
- `nextext/core/openai_cfg.py` — `InferencePipeline` for OpenAI-compatible LLM calls.
- `nextext/core/processing.py` — File I/O and export formatting (CLI).
- `nextext/utils/model_registry.py` — Centralized GPU model residency manager.
- `nextext/utils/mappings/` — JSON config files for Whisper/spaCy model names, language codes.
- `nextext/utils/prompts/` — LLM prompt templates (system, translation, summary).

See `AGENTS.md` for detailed agent documentation including I/O contracts and how to add new agents.

## Environment

Key env vars (see `.env.example`):

- `INFERENCE_PROVIDER` — `ollama` (default), `vllm`, or `openai`. Selects the translation prompt format: `ollama`/`openai` use the templated prompt in `nextext/utils/prompts/translation.txt`, while `vllm` sends a single-user-message delimiter format (`<<<source>>>...<<<target>>>...<<<text>>>...`) required by `Infomaniak-AI/vllm-translategemma-4b-it`.
- `HF_HUB_TOKEN` — required for diarization models
- `OPENAI_API_KEY`, `OPENAI_API_BASE` — OpenAI-compatible endpoint credentials; shared across translation, summarization, and hate-speech detection, so in `vllm` mode the LiteLLM proxy must expose both `TEXT_MODEL` and `TRANSLATION_MODEL` on the same endpoint.
- `TEXT_MODEL`, `TRANSLATION_MODEL` — LLM model names
- `OLLAMA_THINK` — tri-state default for the Ollama `think` request field forwarded by `InferencePipeline.call_model` via `extra_body`. Accepts `1`/`true`/`yes`/`on` (enable), `0`/`false`/`no`/`off` (disable), or unset (omit field, model default). Honoured by Ollama-hosted reasoning models such as Qwen3; a no-op for `vllm`/`openai` providers. Per-call `think=` overrides the env default.
- `NEXTEXT_OFFLINE=1` — offline mode (skip model downloads)
- `MODEL_RESIDENCY_STRATEGY` — `offload` (default) or `evict`. Controls how the registry releases GPU models between files. Per-model overrides: `MODEL_RESIDENCY_GLINER`, `MODEL_RESIDENCY_WHISPER_TURBO`, `MODEL_RESIDENCY_WHISPER_LARGE`, `MODEL_RESIDENCY_DIARIZATION`.
- `BACKEND_HOST` (frontend only) — Backend root URL. Defaults to `http://backend:8000` inside compose; set to `http://localhost:8000` for local dev.
- `BACKEND_PUBLIC_HOST` (frontend only) — Externally reachable backend URL surfaced in UI hints.
- `NEXTEXT_API_HOST` / `NEXTEXT_API_PORT` (backend only) — uvicorn bind address. Defaults to `0.0.0.0:8000`.
- `NEXTEXT_JOB_TTL_SECONDS` (backend only) — Lifetime for completed *ephemeral* jobs before the sweeper evicts them. Persistent jobs are not affected. Defaults to `3600`.
- `NEXTEXT_MAX_UPLOAD_MB` (backend only) — Hard cap on per-upload bytes. Defaults to `8192`.
- `NEXTEXT_DATA_DIR` (backend only) — On-disk root for the SQLite job index and per-job artifact directories. Defaults to `/var/lib/nextext` (the `nextext-data` Docker volume); local dev falls back to `./.nextext-data`.

## Memory management

GPU-resident models (`whisper_turbo`, `whisper_large`, `diarization`, `gliner`) are owned by a process-wide registry in `nextext/utils/model_registry.py`. Callers wrap model use in `with REGISTRY.acquire(name) as model:` so the model is on GPU only for the duration of the block; the registry releases it (offload or evict) on exit. The Streamlit and CLI entry points call `flush_gpu()` between files to reclaim PyTorch allocator reservations. Adding a new GPU model means registering a `ModelSpec` with a `loader` (CPU construction) and `mover` (`.to(device)`).

## Docker

`docker-compose.yml` defines four services across two profiles:

- `backend-cpu` / `backend-cuda` — built from `Dockerfile.backend.{cpu,cuda}`, multi-stage `uv` builds. Run `uvicorn nextext.api.main:app` with a `HEALTHCHECK` against `/api/v1/health`. Reachable only on the `inference-net` network by default; no host port is published. Mounts the `nextext-data` Docker volume at `/var/lib/nextext` for persistent job storage.
- `frontend-cpu` / `frontend-cuda` — built from `Dockerfile.frontend` (single-stage `uv`, `--only-group frontend`). Publishes Streamlit on `${NEXTEXT_HOST_PORT:-8501}` and reaches the backend over the internal network.

Both profiles share `inference-net` with Ollama / vLLM. Run `make volumes` (one-time, creates the external volumes including `nextext-data`) then `make build-cpu && make up-cpu` (or the CUDA equivalents) to bring the full stack up.

## Persistence model

Jobs are ephemeral by default — `JobManager` holds them in memory and the sweeper evicts completed entries after `NEXTEXT_JOB_TTL_SECONDS`. Setting `JobOptions.persist=true` on submission flips that single job to durable storage:

1. `SqliteJobRepository.create()` inserts a row in `<NEXTEXT_DATA_DIR>/jobs.db`, tagged with the caller's `owner_id`.
2. The worker writes artifacts to `<NEXTEXT_DATA_DIR>/jobs/<job_id>/` (Parquet for DataFrames, PNG for the wordcloud, TXT for the summary, JSON for metadata).
3. On backend startup, `JobManager._rehydrate_from_repository()` rebuilds the in-memory states for every row and marks any row still in `queued`/`running` as `interrupted`.
4. The sweeper leaves persistent jobs alone — they live until the owner deletes them.

Postgres-readiness: the persistence surface is the `JobRepository` Protocol in `nextext/api/persistence.py`. A future `PostgresJobRepository` only needs to swap the SQL driver; callers depend on the protocol, not the implementation.

## Commits

- Prefer multiple small topical commits over a single catch-all commit.
- Each commit message should describe a single logical change (refactor, fix, feat, docs, test).
