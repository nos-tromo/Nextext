# Nextext Agents

Nextext breaks the speech-to-insight workflow into specialized agents (self-contained modules) coordinated by the shared pipelines in `nextext/pipeline.py`. Each agent consumes a narrow input, produces a deterministic artifact, and can be toggled on/off from both the CLI (`nextext/cli.py`) and the Streamlit UI (`nextext/app.py`). This document describes every agent, how they interact, and what they expect from the runtime environment.

## Agent Directory

| Agent | Module & entry point | Consumes | Produces | Activated by |
| --- | --- | --- | --- | --- |
| Transcription & Diarization | `nextext/modules/transcription.py` → `WhisperTranscriber`, `transcription_pipeline` | Audio file path, Hugging Face token, Whisper settings | Timestamped transcript `pd.DataFrame` (`start`, `end`, `speaker`, `text`) | Always-on |
| Translation | `nextext/modules/translation.py` → `Translator`, `translation_pipeline` | Transcript `pd.DataFrame`, source and target ISO codes, inference provider | Mutated `pd.DataFrame` translated segment-wise | CLI `--task translate`, Streamlit "Task" switch |
| Word Intelligence | `nextext/modules/words.py` → `WordCounter`, `wordlevel_pipeline` | Transcript text, resolved language | Word counts, entity table, noun sentiment table, graph HTML, word cloud `Figure` | CLI `-w/--words`, Streamlit "Word-level analysis" |
| Summarization | `nextext/pipeline.py` → `summarization_pipeline` + `InferencePipeline` | Full transcript text | Summary string in the configured output language | CLI `-sum/--summarize`, Streamlit "Summarisation" |
| File Export | `nextext/modules/processing.py` → `FileProcessor.write_file_output` | Any agent result | Files in `output/<input-file>/` (`.txt`, `.csv`, `.xlsx`, `.png`) | CLI workflow |
| Inference Service | `nextext/modules/ollama_cfg.py` → `InferencePipeline` | Prompt template + runtime options | Raw string from an OpenAI-compatible chat endpoint | Shared dependency for translation and summary agents |

## Execution Flow

1. **Interface orchestrator** (CLI or Streamlit) collects user options and instantiates `transcription_pipeline`.
2. **Transcription agent** returns a `pd.DataFrame` and detected language code; the orchestration state is updated with the resolved language.
3. **Optional agents** (translation, word-level, summary) fire in the order shown in `nextext/app.py` or `nextext/cli.py`, each mutating or extending the transcript payload.
4. **Outputs** are cached in `st.session_state` for the UI or routed through the `FileProcessor` for CLI batch exports.

> The agents themselves remain stateless; any UI or CLI state is maintained by the orchestrator wrappers.

## Transcription & Diarization Agent

- **Key files:** `nextext/modules/transcription.py`, `transcription_pipeline()` (`nextext/pipeline.py`).
- **Responsibilities:** Load audio, auto-detect language when not provided, run WhisperX transcription, optional diarization (`n_speakers > 1`), and emit a normalized DataFrame used by every downstream agent.
- **Inputs:** `Path` to audio/video, Hugging Face token retrieved through `get_api_key()` (defaults to env var `API_KEY` and persists to `.env`), Whisper model id (`default` resolves via `nextext/utils/mappings/whisper_models.json`), task (`transcribe` or `translate`), target ISO code, optional source code, speaker count.
- **Outputs:** `pd.DataFrame` with `start`, `end`, `speaker`, `text`; detected source language stored in `WhisperTranscriber.src_lang`.
- **Dependencies:** `whisperx`, `torch`, diarization models gated by the Hugging Face token; GPU/MPS detection picks compute dtype automatically.
- **Operational notes:** If diarization is disabled the speaker column defaults to a single speaker; diarization requires extra VRAM and uses `pyannote` models, so ensure they are downloaded before run or the call will raise.

## Translation Agent

- **Key files:** `nextext/modules/translation.py`, `translation_pipeline()` (`nextext/pipeline.py`).
- **Responsibilities:** Detect the transcript language via `langdetect` when needed, prompt TranslateGemma via the shared inference service, and rewrite each transcript segment to the requested target language.
- **Inputs:** Transcript `pd.DataFrame` (only the `text` column is used), target ISO 639-1 code, optional resolved source code from transcription, shared `InferencePipeline`.
- **Outputs:** In-place replacement of the `text` column; `Translator.src_lang` is populated for logging and downstream toggles.
- **Dependencies:** `langdetect`, `pycountry`, Ollama's `translategemma` model by default, or an alternate OpenAI-compatible model when `INFERENCE_PROVIDER=openai`.
- **Operational notes:** Translation is skipped when the resolved source language already equals the target. The default local model is selected from `nextext/utils/mappings/translation_models.json`.

## Word Intelligence Agent

- **Key files:** `nextext/modules/words.py`, `wordlevel_pipeline()` (`nextext/pipeline.py`), spaCy mappings in `nextext/utils/mappings/spacy_models.json`.
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
- **Dependencies:** OpenAI-compatible chat completions via Ollama by default (`OLLAMA_HOST`, defaults to `http://localhost:11434`), or the hosted OpenAI API when `INFERENCE_PROVIDER=openai`; prompt language controlled by `InferencePipeline.out_language`.
- **Operational notes:** The pipeline raises `ValueError` when text is empty; make sure optional agents check for data before calling.

## Inference Service Agent

- **Key files:** `nextext/modules/ollama_cfg.py`, prompts directory `nextext/utils/prompts/`, model mappings `nextext/utils/mappings/ollama_models.json` and `translation_models.json`.
- **Responsibilities:** Construct prompts, pick provider-specific models, create an OpenAI-compatible client, perform provider health checks, and expose `call_model()` to translation and summarization.
- **Inputs:** Prompt keyword (`system`, `summary`), runtime options (temperature, stop tokens, max tokens), provider configuration (`INFERENCE_PROVIDER`, `OLLAMA_HOST`, `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`).
- **Outputs:** Raw string response from the configured chat completion endpoint; `sys_prompt` ensures outputs are emitted in the configured language.
- **Operational notes:** Ollama remains the default provider and is reached through its `/v1/chat/completions` compatibility layer. When `INFERENCE_PROVIDER=openai`, the runtime requires `OPENAI_API_KEY`.

## File Export Agent

- **Key files:** `nextext/modules/processing.py`.
- **Responsibilities:** Create `output/<input-file>/` directories and serialize any agent output (text, list, DataFrame, Matplotlib figure) to disk.
- **Inputs:** Label describing the payload, optional target language suffix.
- **Outputs:** `.txt`, `.csv`, `.xlsx`, or `.png` files depending on type; log statements confirm each save path.
- **Operational notes:** Only the CLI uses `FileProcessor`; the Streamlit UI instead keeps the artifacts in memory and displays/downloads them directly.

## Orchestrators (CLI & Streamlit)

- **Streamlit UI (`nextext/app.py`):** Stores user choices in `st.session_state["opts"]`, uploads files into a temporary path, and runs `_run_pipeline()`. Results are cached per session; UI tabs read from `st.session_state["result"]`.
- **CLI (`nextext/cli.py`):** `parse_arguments()` maps command-line flags to agent toggles. `main()` wires the pipelines sequentially and persists outputs via `FileProcessor`.
- **Shared behaviour:** Both orchestrators guard optional agents behind flags, update the resolved language after transcription, and instantiate a shared `InferencePipeline` when translation or summarization is requested.

## Adding or Modifying Agents

1. **Create the module:** Place new agent code under `nextext/modules/` (or `nextext/pipeline.py` if it is a thin wrapper). Keep the public function signature narrow and return plain Python or pandas objects.
2. **Expose a pipeline hook:** Add a helper function in `nextext/pipeline.py` so both orchestrators can call the agent without duplicating logic.
3. **Wire toggles:** Update `nextext/app.py` (checkbox/radio button + session state) and `nextext/cli.py` (argument flag) so users can opt in/out.
4. **Document configuration:** Extend this file and, if needed, add prompt templates or mapping entries under `nextext/utils/`.
5. **Persist outputs:** Decide whether the result is UI-only or needs a saved artifact; if so, use `FileProcessor.write_file_output()` to follow the existing naming convention.

By keeping each agent isolated and documented here, Nextext can scale its audio-analysis capabilities without creating tight coupling between new features and the UI/CLI front ends.
