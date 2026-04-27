# Nextext Agents

Nextext breaks the speech-to-insight workflow into specialized agents (self-contained modules) coordinated by the shared pipelines in `nextext/pipeline.py`. Each agent consumes a narrow input, produces a deterministic artifact, and can be toggled on/off from both the CLI (`nextext/cli.py`) and the Streamlit UI (`nextext/app.py`). This document describes every agent, how they interact, and what they expect from the runtime environment.

## Agent Directory

| Agent | Module & entry point | Consumes | Produces | Activated by |
| --- | --- | --- | --- | --- |
| Transcription & Diarization | `nextext/core/transcription.py` → `WhisperTranscriber` / `ExternalWhisperTranscriber`, `transcription_pipeline` | Audio file path, Hugging Face token, Whisper settings | Timestamped transcript `pd.DataFrame` (`start`, `end`, `speaker`, `text`) | Always-on |
| Translation | `nextext/core/translation.py` → `Translator`, `translation_pipeline` | Transcript `pd.DataFrame`, source and target ISO codes, inference provider | Mutated `pd.DataFrame` translated segment-wise | CLI `--task translate`, Streamlit "Task" switch |
| Word Intelligence | `nextext/core/words.py` → `WordCounter`, `wordlevel_pipeline` | Transcript text, resolved language | Word counts, entity table, noun sentiment table, graph HTML, word cloud `Figure` | CLI `-w/--words`, Streamlit "Word-level analysis" |
| Summarization | `nextext/pipeline.py` → `summarization_pipeline` + `InferencePipeline` | Full transcript text | Summary string in the configured output language | CLI `-sum/--summarize`, Streamlit "Summarisation" |
| Hate Speech Detection | `nextext/core/hate_speech.py` → `HateSpeechDetector`, `hate_speech_pipeline` | Transcript `pd.DataFrame`, inference provider | List of flagged segment dicts (`hate_speech`, `category`, `confidence`, `reason`, `text`) | CLI `-hs/--hate-speech`, Streamlit "Hate speech detection" |
| File Export | `nextext/core/processing.py` → `FileProcessor.write_file_output` | Any agent result | Files in `output/<input-file>/` (`.txt`, `.csv`, `.xlsx`, `.png`) | CLI workflow |
| Inference Service | `nextext/core/openai_cfg.py` → `InferencePipeline` | Prompt template + runtime options | Raw string from an OpenAI-compatible chat endpoint | Shared dependency for translation, summary, and hate speech agents |

## Execution Flow

1. **Interface orchestrator** (CLI or Streamlit) collects user options and instantiates `transcription_pipeline`.
2. **Transcription agent** returns a `pd.DataFrame` and detected language code; the orchestration state is updated with the resolved language.
3. **Optional agents** (translation, word-level, summary) fire in the order shown in `nextext/app.py` or `nextext/cli.py`, each mutating or extending the transcript payload.
4. **Outputs** are cached in `st.session_state` for the UI or routed through the `FileProcessor` for CLI batch exports.

> The agents themselves remain stateless; any UI or CLI state is maintained by the orchestrator wrappers.

## Transcription & Diarization Agent

- **Key files:** `nextext/core/transcription.py`, `transcription_pipeline()` (`nextext/pipeline.py`).
- **Responsibilities:** Load audio, auto-detect language when not provided, run openai-whisper transcription, optional pyannote-based diarization (`n_speakers > 1`), and emit a normalized DataFrame used by every downstream agent.
- **Inputs:** `Path` to audio/video, task (`transcribe` or `translate`), target ISO code, optional source code, speaker count.
- **Outputs:** `pd.DataFrame` with `start`, `end`, `speaker`, `text`; detected source language stored in `WhisperTranscriber.src_lang`.
- **Providers:** Derived from `INFERENCE_PROVIDER`. `ollama` (default) runs local openai-whisper with hardcoded models (`large-v3-turbo` for transcribe, `large-v3` for translate). `openai` and `vllm` forward the audio to an OpenAI-compatible `/v1/audio/transcriptions` endpoint via `ExternalWhisperTranscriber` (no diarization). The external model defaults to `whisper-1` (openai) or `openai/whisper-large-v3` (vllm) and can be overridden via `WHISPER_MODEL`.
- **Dependencies:** `openai-whisper`, `torch`, `pyannote-audio` (diarization only, gated by Hugging Face token). GPU detection is automatic.
- **Operational notes:** The `large-v3-turbo` model is loaded once and reused for both mel-spectrogram language detection and the transcribe task; the translate task releases it and loads `large-v3`. Diarization assigns speakers via maximum segment overlap from the pyannote timeline. Speaker column is omitted when `n_speakers == 1`.

## Translation Agent

- **Key files:** `nextext/core/translation.py`, `translation_pipeline()` (`nextext/pipeline.py`).
- **Responsibilities:** Detect the transcript language via `langdetect` when needed, prompt TranslateGemma via the shared inference service, and rewrite each transcript segment to the requested target language.
- **Inputs:** Transcript `pd.DataFrame` (only the `text` column is used), target ISO 639-1 code, optional resolved source code from transcription, shared `InferencePipeline`.
- **Outputs:** In-place replacement of the `text` column; `Translator.src_lang` is populated for logging and downstream toggles.
- **Dependencies:** `langdetect`, `pycountry`, plus an OpenAI-compatible chat completions backend selected by `INFERENCE_PROVIDER` (`ollama` by default, `vllm`, or `openai`).
- **Operational notes:** Translation is skipped when the resolved source language already equals the target. The runtime requires an explicit `TRANSLATION_MODEL` environment variable when translation is enabled. The prompt shape depends on `INFERENCE_PROVIDER`: `ollama` and `openai` use the templated prompt in `nextext/utils/prompts/translation.txt` with a system message, while `vllm` sends a single user message in the delimiter format (`<<<source>>>{src}<<<target>>>{trg}<<<text>>>{text}`) required by `Infomaniak-AI/vllm-translategemma-4b-it` — the model card specifies user/assistant roles only, so the system prompt is omitted on that path.

## Word Intelligence Agent

- **Key files:** `nextext/core/words.py`, `wordlevel_pipeline()` (`nextext/pipeline.py`), spaCy mappings in `nextext/utils/mappings/spacy_models.json`.
- **Responsibilities:** Turn transcript text into linguistic diagnostics—top words, named entities, noun sentiment table, noun-verb-adjective network, and a matplotlib word cloud.
- **Inputs:** `pd.DataFrame` with `text`, resolved language code (source for `transcribe`, target for `translate`).
- **Outputs:** Tuple `(word_counts_df, entities_df, noun_sentiment_df, noun_graph_html_path, wordcloud_figure)`.
- **Dependencies:** `spacy`, `pyvis`, `matplotlib`, `camel_tools`, `arabic_reshaper`, `networkx`; fonts loaded via `nextext/utils/font_loader.py` to keep multilingual rendering stable.
- **Operational notes:** Call `text_to_doc()` then `lemmatize_doc()` before counting; spaCy models are downloaded on demand, so make sure the environment can write to the cache or pre-bundle them.

## Summarization Agent

- **Key files:** `summarization_pipeline()` (`nextext/pipeline.py`), prompt template `nextext/utils/prompts/summary.txt`.
- **Responsibilities:** Format the entire transcript as a prompt, apply the system instruction defined in `InferencePipeline`, and return a concise summary limited to 15 sentences.
- **Inputs:** Concatenated transcript string, `InferencePipeline`.
- **Outputs:** Summary string; orchestrators attach it to session state or export it via `FileProcessor`.
- **Dependencies:** OpenAI-compatible chat completions via the backend selected by `INFERENCE_PROVIDER` (Ollama on `http://localhost:11434/v1` by default, a LiteLLM-fronted vLLM stack, or the hosted OpenAI API); prompt language controlled by `InferencePipeline.out_language`. Summarization always uses the templated prompt + system message regardless of provider.
- **Operational notes:** The pipeline raises `ValueError` when text is empty; make sure optional agents check for data before calling.

## Hate Speech Detection Agent

- **Key files:** `nextext/core/hate_speech.py`, `hate_speech_pipeline()` (`nextext/pipeline.py`), prompt template `nextext/utils/prompts/hate_speech.txt`.
- **Responsibilities:** Iterate over each transcript segment, send it to an LLM with the hate speech prompt, and return a list of flagged entries with structured metadata.
- **Inputs:** Transcript `pd.DataFrame` (only the `text` column is used), shared `InferencePipeline`, optional `max_chars` limit (default 2048).
- **Outputs:** `list[dict]` — each entry contains `hate_speech=True`, `category`, `confidence` (`high`/`medium`/`low`), `reason`, and the original `text`. Only flagged segments are included; an empty list means no hate speech was found.
- **Categories:** racism, sexism, homophobia, religious hatred, xenophobia, disability discrimination, none.
- **Dependencies:** OpenAI-compatible chat completions via `InferencePipeline`; same provider configuration as translation and summarization.
- **Operational notes:** The LLM response is parsed as JSON with regex fallback for prose-wrapped output. Parsing failures produce a safe default (`hate_speech=False`). Text is truncated to `max_chars` before sending to the model. CLI results are exported as a CSV via `FileProcessor`; Streamlit displays each flagged entry in a collapsible expander.

## Inference Service Agent

- **Key files:** `nextext/core/openai_cfg.py` and prompts directory `nextext/utils/prompts/`.
- **Responsibilities:** Construct prompts, pick provider-specific models, create an OpenAI-compatible client, perform provider health checks, and expose `call_model()` to translation and summarization.
- **Inputs:** Prompt keyword (`system`, `summary`, `translation`, `hate_speech`), runtime options (temperature, stop tokens, max tokens, `include_system_prompt`), provider configuration (`INFERENCE_PROVIDER`, `OPENAI_API_BASE`, `OPENAI_API_KEY`).
- **Outputs:** Raw string response from the configured chat completion endpoint; `sys_prompt` ensures outputs are emitted in the configured language when the system role is included.
- **Operational notes:** Ollama remains the default provider and is reached through its `/v1/chat/completions` compatibility layer. `INFERENCE_PROVIDER=vllm` targets a LiteLLM-fronted `nos-tromo/vllm-service` stack (LiteLLM dispatches by `model` field, so both `TEXT_MODEL` and `TRANSLATION_MODEL` must be registered on the same endpoint). `INFERENCE_PROVIDER=openai` targets the hosted OpenAI API. `call_model(include_system_prompt=False)` is used by the vLLM translation path only — all other callers keep the default system prompt. `OPENAI_API_KEY` is required for every provider. `OLLAMA_THINK` (tri-state: `1`/`true`/`yes`/`on` to enable, `0`/`false`/`no`/`off` to disable, unset to omit) sets a process-wide default for the Ollama `think` request field; `call_model(think=...)` overrides it per call. Forwarded via `extra_body`, so it is a no-op on vLLM/OpenAI.

## File Export Agent

- **Key files:** `nextext/core/processing.py`.
- **Responsibilities:** Create `output/<input-file>/` directories and serialize any agent output (text, list, DataFrame, Matplotlib figure) to disk.
- **Inputs:** Label describing the payload, optional target language suffix.
- **Outputs:** `.txt`, `.csv`, `.xlsx`, or `.png` files depending on type; log statements confirm each save path.
- **Operational notes:** Only the CLI uses `FileProcessor`; the Streamlit UI instead keeps the artifacts in memory and displays/downloads them directly.

## Orchestrators (CLI & Streamlit)

- **Streamlit UI (`nextext/app.py`):** Stores user choices in `st.session_state["opts"]`, uploads files into a temporary path, and runs `_run_pipeline()`. Results are cached per session; UI tabs read from `st.session_state["result"]`.
- **CLI (`nextext/cli.py`):** `parse_arguments()` maps command-line flags to agent toggles. `main()` wires the pipelines sequentially and persists outputs via `FileProcessor`.
- **Shared behaviour:** Both orchestrators guard optional agents behind flags, update the resolved language after transcription, and instantiate a shared `InferencePipeline` when translation or summarization is requested.

## Adding or Modifying Agents

1. **Create the module:** Place new agent code under `nextext/core/` (or `nextext/pipeline.py` if it is a thin wrapper). Keep the public function signature narrow and return plain Python or pandas objects.
2. **Expose a pipeline hook:** Add a helper function in `nextext/pipeline.py` so both orchestrators can call the agent without duplicating logic.
3. **Wire toggles:** Update `nextext/app.py` (checkbox/radio button + session state) and `nextext/cli.py` (argument flag) so users can opt in/out.
4. **Document configuration:** Extend this file and, if needed, add prompt templates or mapping entries under `nextext/utils/`.
5. **Persist outputs:** Decide whether the result is UI-only or needs a saved artifact; if so, use `FileProcessor.write_file_output()` to follow the existing naming convention.

By keeping each agent isolated and documented here, Nextext can scale its audio-analysis capabilities without creating tight coupling between new features and the UI/CLI front ends.
