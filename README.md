# Nextext 🎙️

**Nextext** is a modular toolkit for transcribing, translating, and analyzing natural language from audio and video files using state-of-the-art machine learning models. It ships as two cooperating services: a FastAPI **backend** that owns the pipeline and GPU model registry, and a Streamlit **frontend** that talks to the backend over HTTP. The same toolkit also exposes a CLI for in-process batch processing.

> This is a personal project that is under heavy development. It could, and likely does, contain bugs, incomplete code,
> or other unintended issues. As such, the software is provided as-is, without warranty of any kind.

## Setup 🛠️

> **Note:** This README describes setup and usage instructions for Linux and macOS environments. Windows users should follow the equivalent steps using the appropriate commands and paths for their system.

### Prerequisites 📋

- [Hugging Face](https://huggingface.co/) account and access token (read)

Without Docker usage:

- [`uv`](https://github.com/astral-sh/uv) for Python version and dependency management
- Any OpenAI-compatible inference provider (e.g. [Ollama](https://ollama.com/)) reachable via `OPENAI_API_BASE`

### Manual installation 📦

Clone the repository and install the dependencies:

```bash
git clone https://github.com/nos-tromo/Nextext.git
cd Nextext
uv sync
```

To enable speaker diarization, accept the user agreement for the following models: [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0) and [`speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1).

### Docker installation 🐳

#### Shared Docker volumes

The compose file uses external cache volumes so model artifacts survive
container recreation:

- `huggingface-cache`
- `nextext-data` (persistent job index + artifacts; survives `docker compose down -v`)
- `nltk-cache`
- `spacy-cache`
- `torch-cache`

The helper script creates them with `docker volume create`:

```bash
make volumes
```

The compose stack loads `.env` into each Nextext container via
`env_file`, so runtime model downloads pick them up.

#### Inference provider

Nextext communicates with any OpenAI-compatible inference provider via `OPENAI_API_BASE` and `OPENAI_API_KEY`. Provider selection is handled entirely through environment variables — no code changes required.

The Nextext compose services join an external Docker network (`inference-net`) so they can reach whichever inference container you deploy on that network. **Create the network and start your inference provider before running the compose stack.**

**Ollama (recommended for local/self-hosted use):**

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

Then configure Nextext to reach it by adding the following to your `.env` file:

```bash
OPENAI_API_BASE=http://ollama:11434/v1
OPENAI_API_KEY=ollama
```

**Hosted OpenAI API:**

```bash
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_KEY=your-key
```

Any other OpenAI-compatible endpoint (vLLM, LiteLLM, etc.) works the same way — set `OPENAI_API_BASE` to the `/v1` endpoint and `OPENAI_API_KEY` to whatever the provider expects.

#### Profile installation

Clone the repository and bring up the stack for CPU or GPU usage (the
CUDA profile requires a CUDA compatible GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)). The profile is read from `PROFILE` in `.env` (default `cpu`); the `Makefile` is the entry point and points Compose at `docker/compose.yaml` for you:

```bash
make build              # build images for the active profile
make up-dev             # CPU by default — starts backend-cpu + frontend-cpu (publishes host ports)
make up-dev PROFILE=cuda  # CUDA — starts backend-cuda + frontend-cuda
```

`make up-dev` layers `docker/compose.override.yaml` so host ports are
published for local development; `make up` (or the base `docker/compose.yaml`
alone) is the production shape and publishes no host ports.

Each profile brings up two containers:

- **Backend** (`backend-cpu` / `backend-cuda`) — FastAPI on port 8000 (internal). Owns the pipeline, model caches, and (optionally) the persistent job index. Exposes `/api/v1/health`, `/api/v1/languages`, `/api/v1/jobs/*`. Not published to the host by default.
- **Frontend** (`frontend-cpu` / `frontend-cuda`) — Streamlit on port 8501. A thin HTTP client over the backend; ships only the `frontend` dependency group (no torch / whisper / pyannote).

By default every job runs ephemerally — uploads stream through RAM and disappear when the sweeper evicts them. Tick **Save results across browser sessions** in the Parameters tab to opt into persistent storage: the backend then writes the job index to SQLite under `/var/lib/nextext` (the `nextext-data` Docker volume) and per-job artifacts to a directory beside it. Persistent jobs survive container restarts. Identity is anonymous — the frontend stamps a UUID4 into the URL (`?owner=<uuid>`) on first visit and the backend uses it to scope rows. Bookmark the page or keep the tab open to keep your saved jobs reachable.

Launch the UI: `http://localhost:8501/`. The frontend reaches the backend via `BACKEND_HOST` (default `http://backend:8000` inside the compose network).

Each build is tagged `nextext-{backend,frontend}-{cpu|cuda}:${NEXTEXT_VERSION}`, where
`NEXTEXT_VERSION` defaults to `latest`. Override it (e.g. for releases)
by exporting `NEXTEXT_VERSION` before running `make` (or a raw
`docker compose -f docker/compose.yaml` invocation).

#### Make shortcuts 🧰

A `Makefile` is the entry point for the Docker workflow — it points
Compose at `docker/compose.yaml` so you don't have to remember the
file path or profile flags. The profile is read from `PROFILE` in `.env`
(default `cpu`); override per-invocation as `make up PROFILE=cuda`:

| Target | Action |
|--------|--------|
| `make network` | Create the external `inference-net` Docker network (one-time per host; idempotent). |
| `make volumes` | Create the external Docker volumes including `nextext-data` for persistent job storage (one-time per host; idempotent). |
| `make build` | Build both backend and frontend images for the active profile. |
| `make up` | Run both services for the active profile in the foreground (production shape — no host ports). |
| `make up-dev` | Same as `make up` but layers `docker/compose.override.yaml` to publish host ports for local development. |
| `make stop` | Stop the active profile's containers. |
| `make logs` | Tail combined logs from backend and frontend. |
| `make bundle` | Build the active profile and write versioned `.tar.gz` archives for offline transfer (see below). |

When invoked through `make`, `NEXTEXT_VERSION` defaults to
`YYYY-MM-DD-<short-sha>` so each build gets a traceable tag. Export
`NEXTEXT_VERSION=…` beforehand to pin a specific version.

#### Offline / air-gapped distribution 📦

To ship Nextext to a host without internet access, run the bundler on a
machine that *does* have access (`make bundle` follows `PROFILE` from
`.env`; override with `make bundle PROFILE=cuda`):

```bash
make bundle   # or: make bundle PROFILE=cuda
```

The script builds the local Nextext image, pulls any externally hosted
images referenced by the compose file, and writes them to two versioned
tarballs in the project root:

- `nextext-built-{profile}-{version}.tar.gz` — locally built Nextext images
- `nextext-pulled-{profile}-{version}.tar.gz` — images pulled from registries

Copy the tarballs (and your `.env` plus the `docker/` directory) to the
target host, load them, and bring up the stack without rebuilding. The
target host runs the production shape — `docker/compose.yaml` without the
dev override — so no host ports are published:

```bash
docker load < nextext-built-cpu-<version>.tar.gz
docker load < nextext-pulled-cpu-<version>.tar.gz   # may be empty for the default compose
export NEXTEXT_VERSION=<version>
docker compose --env-file .env -f docker/compose.yaml --profile cpu up --no-build
```

### Model downloads 📥

Transcription and alignment models used by [WhisperX](https://github.com/m-bain/whisperX/) will be downloaded upon first usage. Some models can be downloaded beforehand:

#### Ollama models 🦙

The following models are recommended and tested for this application (select depending on your hardware setup):

| Purpose | Model |
|---------|-------|
| Summarization / general | [`gemma3:27b-it-qat`](https://ollama.com/library/gemma3), [`gemma3:12b-it-qat`](https://ollama.com/library/gemma3), [`gemma3n:e4b`](https://ollama.com/library/gemma3n) |
| Translation | [`translategemma:27b`](https://ollama.com/library/translategemma), [`translategemma:12b`](https://ollama.com/library/translategemma), [`translategemma:4b`](https://ollama.com/library/translategemma) |

Pull models into the running Ollama container:

```bash
docker exec ollama ollama pull gemma3:12b-it-qat
docker exec ollama ollama pull translategemma:4b
```

Then set the model names in `.env`:

```bash
TEXT_MODEL=gemma3:12b-it-qat
TRANSLATION_MODEL=translategemma:4b
```

#### Local preload command 🌐

```bash
uv run load-models
```

`load-models` preloads Nextext's NLTK resources, configured spaCy
packages, WhisperX speech models, WhisperX alignment models, and the
default diarization pipeline when `HF_HUB_TOKEN` is available. The
legacy alias `uv run load-spacy-models` still works.

#### Offline usage 🚫🌐

In case Nextext is intended to run in a firewalled or offline environment, set the environment variable after completing the model downloads:

```bash
echo 'export HF_HUB_OFFLINE=1' >> .venv/bin/activate
```

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

The backend exposes the same workflow as the UI under `/api/v1/jobs` (multipart upload + SSE event stream + per-artifact downloads) so any HTTP client — `curl`, scripts, other services — can drive the pipeline directly. See `docker/compose.yaml` and `docker/Dockerfile.backend.{cpu,cuda}` for production deployment.

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
