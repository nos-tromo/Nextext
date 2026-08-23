# Nextext

**Nextext** is a modular toolkit for transcribing, translating, and analyzing natural language from audio and video files. All model inference (Whisper transcription, LLM text tasks, GLiNER NER, speaker diarization) runs on **external OpenAI-compatible / HTTP endpoints** — the app itself ships no model weights and needs no GPU. It consists of two cooperating services: a FastAPI **backend** that owns the pipeline and a React SPA **frontend** (served by nginx) that talks to the backend same-origin via `/api/v1`. The same toolkit also exposes a CLI for in-process batch processing.

> This is a personal project that is under heavy development. It could, and likely does, contain bugs, incomplete code,
> or other unintended issues. As such, the software is provided as-is, without warranty of any kind.

> **Note:** This README describes setup and usage instructions for Linux and macOS environments. Windows users should follow the equivalent steps using the appropriate commands and paths for their system.

## Prerequisites

- An OpenAI-compatible inference endpoint reachable via `OPENAI_API_BASE` (e.g. [nos-tromo/vllm-service](https://github.com/nos-tromo/vllm-service) or [Ollama](https://ollama.com/) for the text tasks)
- An endpoint serving Whisper transcription (`/v1/audio/transcriptions`). Most OpenAI-compatible servers provide one; Ollama does not — set `WHISPER_API_BASE` + `WHISPER_MODEL` separately in that case
- _(optional)_ NER (`/gliner`), speaker diarization (`/diarize`), and the VAD speech pre-filter (`/vad`). These default to the central endpoint and can each take a dedicated base URL — see [configuration.md](docs/configuration.md#dedicated-endpoints). Uploads are sent as-is and decoded server-side, so no local `ffmpeg` is required
- Without Docker: [`uv`](https://github.com/astral-sh/uv) for Python version and dependency management

## Quick start

The recommended way to run Nextext is via Docker:

```bash
make network    # create the external inference-net + edge-net (one-time per host)
make volumes    # create the external nltk-cache + spacy-cache volumes (one-time per host)
make build      # build the backend + frontend images
make up-dev     # start both; frontend published on http://localhost:${NEXTEXT_HOST_PORT:-8501}/
```

Open `http://localhost:8501` (or your configured `NEXTEXT_HOST_PORT`) in your browser. nginx serves the React SPA and proxies `/api/v1` to the backend — no separate backend URL is needed in the browser.

`make up-dev` layers `docker/compose.override.yaml` so host ports are published for local development; `make up` (or the base `docker/compose.yaml` alone) is the production shape and publishes no host ports. Both run detached and never build (`--no-build`) — run `make build` first, or `make dev` to build then bring up with host ports. In production the SPA is served under `/nextext/` behind the `edge-plane` gateway; see [configuration.md](docs/configuration.md#production-sub-path).

For local development without Docker, start the backend and the Vite dev server in two terminals:

```bash
# Terminal 1 — FastAPI backend on http://localhost:8000
uv run nextext-api

# Terminal 2 — React Vite dev server (proxies /api/v1 to localhost:8000)
cd frontend && pnpm dev
```

Open `http://localhost:5173` (default Vite port).

The backend exposes the full workflow under `/api/v1/jobs` (multipart upload + SSE event stream + per-artifact downloads) so any HTTP client — `curl`, scripts, other services — can drive the pipeline directly. See `docker/compose.yaml` and `docker/Dockerfile.backend` for production deployment.

## Manual installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/nos-tromo/Nextext.git
cd Nextext
uv sync
```

## Model resources

The backend itself downloads no model weights — inference models live on the
external endpoints. Only the spaCy / NLTK language resources are fetched
locally, and `NEXTEXT_OFFLINE=1` (the default) skips even those:

```bash
NEXTEXT_OFFLINE=0 uv run load-models
```

Preload them on a connected host before shipping to a disconnected one — see
[airgap.md](docs/airgap.md#offline-usage).

## CLI

`uv run nextext-cli -f <file> [ARGS]` runs the same pipeline in-process, with no
backend container — the right tool for very large local files and directory
batch loops. Flags and examples: [cli.md](docs/cli.md).

## Operating

A `Makefile` is the entry point for the Docker workflow — it points Compose at
`docker/compose.yaml` so you don't have to remember the file path. Run
`make help` for the full target list.

## Documentation

Topic-by-topic reference lives in [docs/](docs/README.md):

- [configuration.md](docs/configuration.md) — inference provider, dedicated per-model endpoints, localization, provider setup, upload limits, every other env-var group
- [architecture.md](docs/architecture.md) — the two-service split, the in-memory job and identity model, image tagging
- [airgap.md](docs/airgap.md) — offline usage, `make bundle`, loading on a disconnected host
- [cli.md](docs/cli.md) — `nextext-cli` flags and batch processing

## Pointers

- Inference provider: [nos-tromo/vllm-service](https://github.com/nos-tromo/vllm-service)
- Remaining planned feature work: [issue #149](https://github.com/nos-tromo/Nextext/issues/149)
- Feedback, bugs, and suggestions: <https://github.com/nos-tromo/Nextext/issues>
- Licensed under the Apache License 2.0 — see [LICENSE](LICENSE)
