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

- `POST /jobs` (multipart: `file` + JSON `options`) — queue a new job; returns `{job_id}`.
- `GET /jobs/{id}` — point-in-time snapshot.
- `GET /jobs/{id}/events` — SSE stream of stage transitions.
- `GET /jobs/{id}/artifacts/{name}` — binary download (transcript.csv/xlsx, summary.txt, wordcounts.csv/xlsx, entities.csv/xlsx, wordcloud.png, hate_speech.csv/xlsx, docint.jsonl, archive.zip).
- `DELETE /jobs/{id}` — cleanup.
- `GET /health`, `GET /languages` — meta endpoints.

**Key modules:**

- `nextext/api/main.py` — FastAPI factory, lifespan (boots `JobManager`).
- `nextext/api/jobs.py` — `JobManager`, async worker (single in-flight job via `asyncio.Semaphore(1)`), SSE event broker.
- `nextext/api/routes/` — `health`, `jobs` routers.
- `nextext/api/artifacts.py` — Per-job artifact byte materializers (CSV/XLSX/PNG/JSONL/ZIP).
- `nextext/api/schemas.py` — Pydantic request/response models.
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
- `NEXTEXT_JOB_TTL_SECONDS` (backend only) — Lifetime for completed jobs before the sweeper evicts them. Defaults to `3600`.
- `NEXTEXT_MAX_UPLOAD_MB` (backend only) — Hard cap on per-upload bytes. Defaults to `8192`.

## Memory management

GPU-resident models (`whisper_turbo`, `whisper_large`, `diarization`, `gliner`) are owned by a process-wide registry in `nextext/utils/model_registry.py`. Callers wrap model use in `with REGISTRY.acquire(name) as model:` so the model is on GPU only for the duration of the block; the registry releases it (offload or evict) on exit. The Streamlit and CLI entry points call `flush_gpu()` between files to reclaim PyTorch allocator reservations. Adding a new GPU model means registering a `ModelSpec` with a `loader` (CPU construction) and `mover` (`.to(device)`).

## Docker

`docker-compose.yml` defines four services across two profiles:

- `backend-cpu` / `backend-cuda` — built from `Dockerfile.backend.{cpu,cuda}`, multi-stage `uv` builds. Run `uvicorn nextext.api.main:app` with a `HEALTHCHECK` against `/api/v1/health`. Reachable only on the `inference-net` network by default; no host port is published.
- `frontend-cpu` / `frontend-cuda` — built from `Dockerfile.frontend` (single-stage `uv`, `--only-group frontend`). Publishes Streamlit on `${NEXTEXT_HOST_PORT:-8501}` and reaches the backend over the internal network.

Both profiles share `inference-net` with Ollama / vLLM. Run `make build-cpu && make up-cpu` (or the CUDA equivalents) to bring the full stack up.

## Commits

- Prefer multiple small topical commits over a single catch-all commit.
- Each commit message should describe a single logical change (refactor, fix, feat, docs, test).
