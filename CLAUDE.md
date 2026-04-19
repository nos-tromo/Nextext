# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nextext is a modular audio analysis toolkit that transcribes, translates, and analyzes natural language from audio/video files. It uses openai-whisper for transcription, pyannote-audio for diarization, GLiNER for named-entity recognition, spaCy/NLTK for word-level NLP, and LLMs (Ollama, vLLM, or OpenAI-compatible endpoints) for translation, summarization, and hate-speech detection.

## Project Context

- WhisperX has been removed from this project; use openai-whisper + pyannote.
- Target torch install is split by extras (cpu/cuda) via conflicts in pyproject.toml.
- Docker base image is pinned to `python:3.11.12-slim-bookworm` across all Dockerfiles.

## Commands

```bash
# Install dependencies
uv sync                    # production deps
uv sync --group dev        # include dev deps (pytest, ruff, mypy, pre-commit)

# Run the app
uv run nextext             # Streamlit web UI on port 8501
uv run nextext-cli -f <file> [args]  # CLI mode

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
- Python 3.11 is the target; prefer explicit types and distinct variable names across branches to satisfy mypy.

## Architecture

**Agent-based design:** Each feature is a stateless agent (module) with narrow input/output. Orchestrators (Streamlit UI or CLI) manage state and compose agents via shared pipelines.

**Pipeline flow:**

1. **Transcription** (always-on) → openai-whisper transcription + optional pyannote diarization → `pd.DataFrame`
2. **Translation** (optional) → LLM-based segment translation via `InferencePipeline`
3. **Word-level analysis** (optional) → word counts, GLiNER named entities, word clouds
4. **Summarization** (optional) → LLM summary via `InferencePipeline`
5. **Hate-speech detection** (optional) → per-segment LLM classification
6. **File export** (CLI) / **ZIP download** (Streamlit) → `.txt`, `.csv`, `.xlsx`, `.png`

**Key modules:**

- `nextext/app.py` — Streamlit UI entry point, session state orchestration
- `nextext/cli.py` — CLI entry point (argparse), single-file processing
- `nextext/pipeline.py` — Shared pipeline functions connecting all agents
- `nextext/core/transcription.py` — openai-whisper transcription & pyannote diarization
- `nextext/core/translation.py` — LLM translation with prompt templates
- `nextext/core/words.py` — NLP word-level analysis (spaCy + GLiNER NER)
- `nextext/core/hate_speech.py` — LLM-based hate-speech detection
- `nextext/core/openai_cfg.py` — `InferencePipeline` for OpenAI-compatible LLM calls
- `nextext/core/processing.py` — File I/O and export formatting
- `nextext/utils/model_registry.py` — Centralized GPU model residency manager
- `nextext/utils/mappings/` — JSON config files for Whisper/spaCy model names, language codes
- `nextext/utils/prompts/` — LLM prompt templates (system, translation, summary)

See `AGENTS.md` for detailed agent documentation including I/O contracts and how to add new agents.

## Environment

Key env vars (see `.env.example`):

- `INFERENCE_PROVIDER` — `ollama` (default), `vllm`, or `openai`. Selects the translation prompt format: `ollama`/`openai` use the templated prompt in `nextext/utils/prompts/translation.txt`, while `vllm` sends a single-user-message delimiter format (`<<<source>>>...<<<target>>>...<<<text>>>...`) required by `Infomaniak-AI/vllm-translategemma-4b-it`.
- `HF_HUB_TOKEN` — required for diarization models
- `OPENAI_API_KEY`, `OPENAI_API_BASE` — OpenAI-compatible endpoint credentials; shared across translation, summarization, and hate-speech detection, so in `vllm` mode the LiteLLM proxy must expose both `TEXT_MODEL` and `TRANSLATION_MODEL` on the same endpoint.
- `TEXT_MODEL`, `TRANSLATION_MODEL` — LLM model names
- `NEXTEXT_OFFLINE=1` — offline mode (skip model downloads)
- `MODEL_RESIDENCY_STRATEGY` — `offload` (default) or `evict`. Controls how the registry releases GPU models between files. Per-model overrides: `MODEL_RESIDENCY_GLINER`, `MODEL_RESIDENCY_WHISPER_TURBO`, `MODEL_RESIDENCY_WHISPER_LARGE`, `MODEL_RESIDENCY_DIARIZATION`.

## Memory management

GPU-resident models (`whisper_turbo`, `whisper_large`, `diarization`, `gliner`) are owned by a process-wide registry in `nextext/utils/model_registry.py`. Callers wrap model use in `with REGISTRY.acquire(name) as model:` so the model is on GPU only for the duration of the block; the registry releases it (offload or evict) on exit. The Streamlit and CLI entry points call `flush_gpu()` between files to reclaim PyTorch allocator reservations. Adding a new GPU model means registering a `ModelSpec` with a `loader` (CPU construction) and `mover` (`.to(device)`).

## Docker

`docker-compose.yml` defines CPU and CUDA profiles for both Nextext and Ollama services. Multi-stage Dockerfiles (`Dockerfile.cpu`, `Dockerfile.cuda`) use `uv` for dependency installation.

## Commits

- Prefer multiple small topical commits over a single catch-all commit.
- Each commit message should describe a single logical change (refactor, fix, feat, docs, test).
