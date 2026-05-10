# Nextext 🎙️

**Nextext** is a modular toolkit for transcribing, translating, and analyzing natural language from audio and video files using state-of-the-art machine learning models. Designed for flexibility, it supports both a user-friendly Streamlit web interface and command-line operation. The results are compiled into structured output files featuring transcriptions, summaries, and word-level statistics.

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

Clone the repository and run either for CPU or GPU usage (requires a CUDA compatible GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for the CUDA profile):

```bash
docker compose --profile cpu up  # CPU
docker compose --profile cuda up  # CUDA
```

Launch the app: `http://localhost:8501/`

Each build is tagged `nextext-{cpu|cuda}:${NEXTEXT_VERSION}`, where
`NEXTEXT_VERSION` defaults to `latest`. Override it (e.g. for releases)
by exporting `NEXTEXT_VERSION` before running `docker compose` or `make`.

#### Make shortcuts 🧰

A `Makefile` wraps the most common compose flows so you don't have to
remember profile flags:

| Target | Action |
|--------|--------|
| `make volumes` | Create the external Docker volumes (one-time per host; idempotent). |
| `make build-cpu` / `make build-cuda` | Build the chosen profile's image. |
| `make up-cpu` / `make up-cuda` | Build and run the chosen profile in the foreground. |
| `make no-build` / `make no-build-cuda` | Run the stack from already-built (or freshly loaded) images, skipping the build step. |
| `make bundle-cpu` / `make bundle-cuda` | Build a profile and write versioned `.tar.gz` archives for offline transfer (see below). |

When invoked through `make`, `NEXTEXT_VERSION` defaults to
`YYYY-MM-DD-<short-sha>` so each build gets a traceable tag. Export
`NEXTEXT_VERSION=…` beforehand to pin a specific version.

#### Offline / air-gapped distribution 📦

To ship Nextext to a host without internet access, run the bundler on a
machine that *does* have access:

```bash
make bundle-cpu   # or: make bundle-cuda
```

The script builds the local Nextext image, pulls any externally hosted
images referenced by the compose file, and writes them to two versioned
tarballs in the project root:

- `nextext-built-{profile}-{version}.tar.gz` — locally built Nextext images
- `nextext-pulled-{profile}-{version}.tar.gz` — images pulled from registries

Copy the tarballs (and your `.env`) to the target host, load them, and
bring up the stack without rebuilding:

```bash
docker load < nextext-built-cpu-<version>.tar.gz
docker load < nextext-pulled-cpu-<version>.tar.gz   # may be empty for the default compose
export NEXTEXT_VERSION=<version>
make no-build   # or: make no-build-cuda
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

To launch the Streamlit web interface, run the following command from the project root:

```bash
uv run nextext
```

This will start the app locally and provide a URL (typically `http://localhost:8501`) in your terminal. Open this URL in your browser to access the Nextext interface.

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
