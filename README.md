# Nextext 🎙️

**Nextext** is a modular toolkit for transcribing, translating, and analyzing natural language from audio and video files using state-of-the-art machine learning models. Designed for flexibility, it supports both a user-friendly Streamlit web interface and command-line operation. The results are compiled into structured output files featuring transcriptions, summaries, statistics, topic modeling, and sentiment analysis, making it ideal for comprehensive audio and text analysis workflows.

> This is a personal project that is under heavy development. It could, and likely does, contain bugs, incomplete code,
> or other unintended issues. As such, the software is provided as-is, without warranty of any kind.

## Setup 🛠️

> **Note:** This README describes setup and usage instructions for Linux and macOS environments. Windows users should follow the equivalent steps using the appropriate commands and paths for their system.

### Prerequisites 📋

- [`uv`](https://github.com/astral-sh/uv) for Python version and dependency management
- [`Ollama`](https://ollama.com/) for local inference
- [Hugging Face](https://huggingface.co/) account and access token (read)

### Manual installation 📦

Clone the repository, create the virtual environment and install the dependencies:

```bash
uv venv
uv pip install .
uv sync
```

To enable speaker diarization, accept the user agreement for the following models: [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0) and [`speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1).

### Model downloads 📥

Nextext requires downloading some models:

#### Hugging Face 🤗

- [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo)
- [`openai/whisper-large-v2`](https://huggingface.co/openai/whisper-large-v2)
- [`openai/whisper-small`](https://huggingface.co/openai/whisper-small)
- [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [`google/madlad400-3b-mt`](https://huggingface.co/google/madlad400-3b-mt)
- [`textdetox/xlmr-large-toxicity-classifier`](https://huggingface.co/textdetox/xlmr-large-toxicity-classifier)

#### Ollama 🦙

The following models are recommended and tested for this application (select depending on your hardware setup):

- [`gemma3:4b-it-qat`](https://ollama.com/library/gemma3)
- [`gemma3:12b-it-qat`](https://ollama.com/library/gemma3)
- [`gemma3:27b-it-qat`](https://ollama.com/library/gemma3)

To configure the app's default models, edit the selector located at `nextext/utils/ollama_models.json`.

#### Other language and tokenization models 🌐

```bash
# nltk
uv run python -c "import nltk; nltk.download('stopwords')"
# spaCy
uv run python -m nextext.utils.spacy_model_loader
```

#### Offline usage 🚫🌐

In case Nextext is intended to run in a firewalled or offline environment, set the environment variable after completing the model downloads:

```bash
echo 'export HF_HUB_OFFLINE=1' >> .venv/bin/activate
```

For Windows, add `$env:HF_HUB_OFFLINE = "1"` to `.venv/Scripts/Activate.ps1` above the line `# SIG # Begin`.

### Docker installation 🐳

Select whether to use the CPU or GPU (requires a CUDA compatible GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) set up):

```bash
docker compose --profile cpu up
docker compose --profile gpu up
```

Launch the app: `http://localhost:8501/`

> **Important:** The Docker installation requires that all models listed above are downloaded in advance and that Ollama is installed on your system.

## Usage 🚀

### Streamlit 🌈

To launch the Streamlit web interface, run the following command from the project root:

```bash
uv run streamlit run app.py
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

Running `uv run python -m nextext.cli [ARGS]` from the command line supports the following arguments:

```bash
-h, --help            show this help message and exit
-f, --file            Specify the file path and name of the audio file to be transcribed.
-sl, --src-lang       Specify the language code (ISO 639-1) of the source audio (default: 'default').
-tl, --trg-lang       Specify the language code (ISO 639-1) of the target language (default: 'de').
-m, --model           Specify the model size for Whisper (default: 'default' = 'turbo').
-t, --task            Specify the task to perform: 'transcribe' (default), or 'translate'.
-s, --speakers        Specify the maximum number of speakers for diarization (default: 1).
-w, --words           Show most frequently used words (default: False).
-tm, --topics         Enable topic modeling analysis (default: False).
-sum, --summarize     Additional text and topic summarization (default: False).
-tox, --toxicity      Enable toxicity analysis (default: False).
-o, --output          Specify the output directory (default: output).
-F, --full-analysis   Enable full analysis, equivalent to using -w -tm -sum -tox (default: False).
```

In CLI mode, you can let Nextext iterate over a directory to batch process files:

```bash
for file in path/to/your/directory/*; do
    uv run python -m nextext.cli -f $file [ARGS]
done
```

## ...to be continued ⏳

- ~~🐳 Dockerize the application~~
- 🛠️ Refactor proper logging and error handling
- 🧪 Improve overall toxicity classification quality
- 🎨 Polish Streamlit frontend
- 🤖 Integrate LLM chatbot into UI
- 📊 Add comprehensive report output
- 🚫 Fix offline usage
- 👥 Implement multi-user access

## Feedback 💬

I hope you find Nextext to be a valuable tool for analysis. If you have any feedback or suggestions on how to improve Nextext, please let me know.
