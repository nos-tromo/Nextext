# Nextext 🎙️

**Nextext** is a modular toolkit for transcribing, translating, and analyzing natural language from audio and video files. All model inference (Whisper transcription, LLM text tasks, GLiNER NER, speaker diarization) runs on **external OpenAI-compatible / HTTP endpoints** — the app itself ships no model weights and needs no GPU. It consists of two cooperating services: a FastAPI **backend** that owns the pipeline and a Streamlit **frontend** that talks to the backend over HTTP. The same toolkit also exposes a CLI for in-process batch processing.

> This is a personal project that is under heavy development. It could, and likely does, contain bugs, incomplete code,
> or other unintended issues. As such, the software is provided as-is, without warranty of any kind.

## Setup 🛠️

> **Note:** This README describes setup and usage instructions for Linux and macOS environments. Windows users should follow the equivalent steps using the appropriate commands and paths for their system.

### Prerequisites 📋

- An OpenAI-compatible inference endpoint reachable via `OPENAI_API_BASE` (e.g. [nos-tromo/vllm-service](https://github.com/nos-tromo/vllm-service) or [Ollama](https://ollama.com/) for the text tasks)
- An endpoint serving Whisper transcription (`/v1/audio/transcriptions`). Most OpenAI-compatible servers provide one; Ollama does not — set `WHISPER_API_BASE` + `WHISPER_MODEL` separately in that case
- _(optional)_ NER (`/gliner`), speaker diarization (`/diarize`), and the VAD speech pre-filter (`/vad`). These default to the central endpoint (one trailing `/v1` stripped) and otherwise take a dedicated `NER_API_BASE` / `DIARIZE_API_BASE` / `VAD_API_BASE`. NER and diarization run only when a job requests entities / multiple speakers; the VAD guard runs on every transcription and is switched off with `VAD_API_BASE=off`. Uploads are sent as-is and decoded server-side, so no local `ffmpeg` is required.

Without Docker usage:

- [`uv`](https://github.com/astral-sh/uv) for Python version and dependency management

### Manual installation 📦

Clone the repository and install the dependencies:

```bash
git clone https://github.com/nos-tromo/Nextext.git
cd Nextext
uv sync
```

Speaker diarization runs out-of-process against an HTTP `/diarize` service. It uses the central endpoint by default, or set `DIARIZE_API_BASE` to a dedicated service root; the gated-model agreements and any Hugging Face token live on the service side, not in Nextext.

### Docker installation 🐳

#### Shared Docker volumes

The compose file uses external cache volumes so the spaCy / NLTK language
resources survive container recreation (no model weights are cached —
inference runs on external endpoints):

- `nltk-cache`
- `spacy-cache`

The helper script creates them with `docker volume create`:

```bash
make volumes
```

The compose stack loads `.env` into each Nextext container via
`env_file`, so runtime model downloads pick them up.

#### Inference provider

Nextext communicates with any OpenAI-compatible inference provider via `OPENAI_API_BASE` and `OPENAI_API_KEY`. Provider selection is handled entirely through environment variables — no code changes required.

Every model class can also be re-pointed at a **dedicated endpoint**, falling back to the central pair when unset:

| Model | Dedicated env vars | Endpoint shape |
|-------|--------------------|----------------|
| Whisper transcription | `WHISPER_API_BASE` / `WHISPER_API_KEY` / `WHISPER_MODEL` | OpenAI SDK base incl. `/v1` |
| GLiNER NER | `NER_API_BASE` (+ `NER_TIMEOUT`) | service root, `POST {base}/gliner` |
| Speaker diarization | `DIARIZE_API_BASE` (+ `DIARIZE_TIMEOUT`) | service root, `POST {base}/diarize` |
| VAD speech guard | `VAD_API_BASE` (+ `VAD_TIMEOUT`) | service root, `POST {base}/vad` |

The NER, diarization, and VAD services speak a plain service root rather than the OpenAI `/v1` shape, so the central fallback strips one trailing `/v1` from `OPENAI_API_BASE` before appending the service path (`http://vllm-router:4000/v1` → `http://vllm-router:4000/gliner`). Whisper, which speaks `/v1`, uses `OPENAI_API_BASE` verbatim. The three non-OpenAI services reuse `OPENAI_API_KEY` as their bearer token; none takes a dedicated key.

NER and diarization issue a request only when a job asks for entities or more than one speaker, so they need no off switch. The VAD guard runs ahead of every transcription (fail-open: an unreachable service transcribes anyway); switch it off with `VAD_API_BASE=off` (also `false` / `no` / `0`).

> **Diarization** and **VAD** reach plain `POST /diarize` and `POST /vad` services (multipart `file` + form fields → JSON). Point `DIARIZE_API_BASE` / `VAD_API_BASE` — or the central endpoint — at a service implementing them; the full request/response contracts live in `nextext/core/diarization.py` and `nextext/core/vad.py`.

The Nextext compose services join an external Docker network (`inference-net`) so they can reach whichever inference container you deploy on that network. **Create the network and start your inference provider before running the compose stack.**

**Ollama (text tasks only — needs a separate Whisper endpoint):**

```bash
# Create Docker network and persistent cache
docker network create inference-net

# Run the Ollama service
docker run -d \
  --network inference-net \
  --name ollama \
  --gpus all \
  -v ollama-cache:/root/.ollama \
  -p 11434:11434 \
  ollama/ollama:0.20.2
```

Then configure Nextext to reach it by adding the following to your `.env` file (Ollama serves no transcription API, so Whisper needs an explicit dedicated endpoint):

```bash
OPENAI_API_BASE=http://ollama:11434/v1
OPENAI_API_KEY=ollama
WHISPER_API_BASE=http://<your-whisper-host>:8000/v1
WHISPER_MODEL=openai/whisper-large-v3
```

**Hosted OpenAI API:**

```bash
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_KEY=your-key
```

Any other OpenAI-compatible endpoint (vLLM, LiteLLM, etc.) works the same way — set `OPENAI_API_BASE` to the `/v1` endpoint and `OPENAI_API_KEY` to whatever the provider expects.

#### Stack installation

Clone the repository and bring up the stack — a single CPU-only image pair,
no GPU or profile selection needed (the `Makefile` is the entry point and
points Compose at `docker/compose.yaml` for you):

```bash
make build              # build the backend + frontend images
make up-dev             # starts backend + frontend (publishes the frontend port)
```

`make up-dev` layers `docker/compose.override.yaml` so host ports are
published for local development; `make up` (or the base `docker/compose.yaml`
alone) is the production shape and publishes no host ports.

The stack brings up two containers:

- **Backend** (`backend`) — FastAPI on port 8000 (internal). Owns the pipeline, the HTTP inference clients, and the in-memory job store. Exposes `/api/v1/health`, `/api/v1/languages`, `/api/v1/jobs/*`. Not published to the host by default.
- **Frontend** (`frontend`) — Streamlit on port 8501. A thin HTTP client over the backend; ships only the `frontend` dependency group.

Jobs live only in memory — there is no durable storage and no TTL, so a long-running job is never cut off and is retained until you delete it or the backend restarts. Identity is anonymous: the frontend mints a per-browser id and stamps it into the URL (`?owner=<id>`) on first visit, sending it to the backend as the trusted identity header (`X-Auth-User` by default) to scope your jobs. Because that id survives a refresh, reloading the page mid-run re-discovers your jobs and resumes the live progress view; closing the tab and reopening the bare host starts a fresh identity. Developers calling the API directly can skip the header and set `NEXTEXT_DEFAULT_IDENTITY` instead. There is no authentication — the backend trusts whoever can reach `inference-net`.

Launch the UI: `http://localhost:8501/`. The frontend reaches the backend via `BACKEND_HOST` (default `http://backend:8000` inside the compose network).

Each build is tagged `nextext-{backend,frontend}:${NEXTEXT_VERSION}`, where
`NEXTEXT_VERSION` defaults to `latest`. Override it (e.g. for releases)
by exporting `NEXTEXT_VERSION` before running `make` (or a raw
`docker compose -f docker/compose.yaml` invocation).

#### Make shortcuts 🧰

A `Makefile` is the entry point for the Docker workflow — it points
Compose at `docker/compose.yaml` so you don't have to remember the
file path:

| Target | Action |
|--------|--------|
| `make network` | Create the external `inference-net` Docker network (one-time per host; idempotent). |
| `make volumes` | Create the external Docker volumes (one-time per host; idempotent). |
| `make build` | Build the backend and frontend images. |
| `make up` | Run both services in the foreground (production shape — no host ports). |
| `make up-dev` | Same as `make up` but layers `docker/compose.override.yaml` to publish host ports for local development. |
| `make stop` | Stop the containers. |
| `make logs` | Tail combined logs from backend and frontend. |
| `make bundle` | Build the images and write versioned `.tar.gz` archives for offline transfer (see below). |

When invoked through `make`, `NEXTEXT_VERSION` defaults to
`YYYY-MM-DD-<short-sha>` so each build gets a traceable tag. Export
`NEXTEXT_VERSION=…` beforehand to pin a specific version.

#### Offline / air-gapped distribution 📦

To ship Nextext to a host without internet access, run the bundler on a
machine that *does* have access:

```bash
make bundle
```

The script builds the local Nextext image, pulls any externally hosted
images referenced by the compose file, and writes them to two versioned
tarballs in the project root:

- `nextext-built-{version}.tar.gz` — locally built Nextext images
- `nextext-pulled-{version}.tar.gz` — images pulled from registries

Copy the tarballs (and your `.env` plus the `docker/` directory) to the
target host, load them, and bring up the stack without rebuilding. The
target host runs the production shape — `docker/compose.yaml` without the
dev override — so no host ports are published:

```bash
docker load < nextext-built-<version>.tar.gz
docker load < nextext-pulled-<version>.tar.gz   # may be empty for the default compose
export NEXTEXT_VERSION=<version>
docker compose --env-file .env -f docker/compose.yaml up --no-build
```

### Model downloads 📥

The backend itself downloads no model weights — inference models live on the
external endpoints. Only the spaCy / NLTK language resources are fetched
locally (see the preload command below).

#### Ollama models 🦙

The following models are recommended and tested for this application (select depending on your hardware setup):

| Purpose | Model |
|---------|-------|
| Summarization / general | [`gemma3:27b-it-qat`](https://ollama.com/library/gemma3), [`gemma3:12b-it-qat`](https://ollama.com/library/gemma3), [`gemma3n:e4b`](https://ollama.com/library/gemma3n) |
| Translation | [`translategemma:27b`](https://ollama.com/library/translategemma), [`translategemma:12b`](https://ollama.com/library/translategemma), [`translategemma:4b`](https://ollama.com/library/translategemma) |

Pull models into the running Ollama container:

```bash
docker exec ollama ollama pull gemma3:12b-it-qat
```

Then set the model names in `.env`:

```bash
TEXT_MODEL=gemma3:12b-it-qat
```

#### Local preload command 🌐

```bash
NEXTEXT_OFFLINE=0 uv run load-models
```

`load-models` preloads Nextext's NLTK resources and the configured spaCy
packages — the only assets fetched locally; all model inference is remote.
The legacy alias `uv run load-spacy-models` still works.

#### Offline usage 🚫🌐

`NEXTEXT_OFFLINE=1` is the default: spaCy / NLTK downloads are skipped and an
uncached spaCy model raises an actionable error instead of attempting a
doomed download. Preload the caches on a connected host (command above) or
ship the `nltk-cache` / `spacy-cache` volumes alongside the image bundle.

## Usage 🚀

### Streamlit 🌈

The Streamlit frontend talks to the FastAPI backend over HTTP. Start both in two terminals:

```bash
# Terminal 1 — FastAPI backend on http://localhost:8000
uv run nextext-api

# Terminal 2 — Streamlit frontend on http://localhost:8501
BACKEND_HOST=http://localhost:8000 uv run nextext
```

Open `http://localhost:8501` in your browser. The `BACKEND_HOST` env var defaults to the Docker-internal alias `http://backend:8000`; override it as shown when running outside compose.

The backend exposes the same workflow as the UI under `/api/v1/jobs` (multipart upload + SSE event stream + per-artifact downloads) so any HTTP client — `curl`, scripts, other services — can drive the pipeline directly. See `docker/compose.yaml` and `docker/Dockerfile.backend` for production deployment.

#### Increasing file upload size limit 📂

By default, Streamlit limits the maximum file upload size to 200MB. To increase this limit, modify `~/.streamlit/config.toml` (you might have to create `config.toml` first). Add or update the following line under the `[server]` section:

```toml
[server]
maxUploadSize = 1024  # Set this value to the required limit in megabytes
```

This example sets the limit to 1024MB (1GB). Restart the Streamlit app after making this change for the new limit to take effect.

### CLI 💻

Running `uv run nextext-cli [ARGS]` from the command line supports the following arguments:

```bash
-h, --help            show this help message and exit
-f, --file            Specify the file path and name of the audio file to be transcribed.
-sl, --src-lang       Specify the language code (ISO 639-1) of the source audio (default: None).
-tl, --trg-lang       Specify the language code (ISO 639-1) of the target language (default: 'de').
-t, --task            Specify the task to perform: 'transcribe' (default), or 'translate'.
-s, --speakers        Specify the maximum number of speakers for diarization (default: 1).
-w, --words           Show most frequently used words (default: False).
-sum, --summarize     Additional transcript summarization (default: False).
-o, --output          Specify the output directory (default: output).
-F, --full-analysis   Enable full analysis, equivalent to using -w -sum (default: False).
```

In CLI mode, you can let Nextext iterate over a directory to batch process files:

```bash
for file in path/to/your/directory/*; do
    uv run nextext-cli -f $file [ARGS]
done
```

## ...to be continued ⏳

- ~~🐳 Dockerize the application~~
- 🛠️ Refactor proper logging and error handling
- 🎨 Polish Streamlit frontend
- 🤖 Integrate LLM chatbot into UI
- 📊 Add comprehensive report output
- 🚫 Fix offline usage
- 👥 Implement multi-user access

## Feedback 💬

I hope you find Nextext to be a valuable tool for analysis. If you have any feedback or suggestions on how to improve Nextext, please let me know.
