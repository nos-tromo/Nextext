# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nextext is a modular audio analysis toolkit that transcribes, translates, and analyzes natural language from audio/video files. It uses WhisperX for transcription, spaCy/NLTK for NLP, and LLMs (Ollama or OpenAI-compatible) for translation and summarization.

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

## Architecture

**Agent-based design:** Each feature is a stateless agent (module) with narrow input/output. Orchestrators (Streamlit UI or CLI) manage state and compose agents via shared pipelines.

**Pipeline flow:**
1. **Transcription** (always-on) → WhisperX transcription + optional diarization → `pd.DataFrame`
2. **Translation** (optional) → LLM-based segment translation via `InferencePipeline`
3. **Word-level analysis** (optional) → word counts, named entities, sentiment, graphs, word clouds
4. **Summarization** (optional) → LLM summary via `InferencePipeline`
5. **File export** (CLI only) → `.txt`, `.csv`, `.xlsx`, `.png` to `output/<input-file>/`

**Key modules:**
- `nextext/app.py` — Streamlit UI entry point, session state orchestration
- `nextext/cli.py` — CLI entry point (Typer), argument parsing, batch processing
- `nextext/pipeline.py` — Shared pipeline functions connecting all agents
- `nextext/modules/transcription.py` — WhisperX transcription & speaker diarization
- `nextext/modules/translation.py` — LLM translation with prompt templates
- `nextext/modules/words.py` — NLP word-level analysis (spaCy, NLTK)
- `nextext/modules/openai_cfg.py` — `InferencePipeline` for OpenAI-compatible LLM calls
- `nextext/modules/processing.py` — File I/O and export formatting
- `nextext/utils/mappings/` — JSON config files for Whisper/spaCy model names, language codes
- `nextext/utils/prompts/` — LLM prompt templates (system, translation, summary)

See `AGENTS.md` for detailed agent documentation including I/O contracts and how to add new agents.

## Environment

Key env vars (see `.env.example`):
- `INFERENCE_PROVIDER` — `ollama` or `openai`
- `HF_HUB_TOKEN` — required for diarization models
- `OPENAI_API_KEY`, `OPENAI_API_BASE` — for OpenAI provider
- `TEXT_MODEL`, `TRANSLATION_MODEL` — LLM model names
- `NEXTEXT_OFFLINE=1` — offline mode (skip model downloads)

## Testing

Tests are in `tests/` using pytest with monkeypatch fixtures for mocking ML models and inference. Tests simulate Docker detection and environment configuration. No GPU or model downloads required for tests.

## Docker

`docker-compose.yml` defines CPU and CUDA profiles for both Nextext and Ollama services. Multi-stage Dockerfiles (`Dockerfile.cpu`, `Dockerfile.cuda`) use `uv` for dependency installation.
