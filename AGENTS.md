# Nextext Agents

Nextext breaks the speech-to-insight workflow into specialized agents (self-contained modules) coordinated by the shared pipelines in `nextext/pipeline.py`. Each agent consumes a narrow input, produces a deterministic artifact, and can be toggled on/off from both the CLI (`nextext/cli.py`) and the Streamlit UI (`nextext/app.py`). This document describes every agent, how they interact, and what they expect from the runtime environment.

## Agent Directory

| Agent | Module & entry point | Consumes | Produces | Activated by |
| --- | --- | --- | --- | --- |
| Transcription & Diarization | `nextext/modules/transcription.py` → `WhisperTranscriber`, `transcription_pipeline` | Audio file path, Hugging Face token, Whisper settings | Timestamped transcript `pd.DataFrame` (`start`, `end`, `speaker`, `text`) | Always-on |
| Translation | `nextext/modules/translation.py` → `Translator`, `translation_pipeline` | Transcript `pd.DataFrame`, target ISO code | Mutated `pd.DataFrame` translated sentence-wise | CLI `--task translate`, Streamlit "Task" switch |
| Word Intelligence | `nextext/modules/words.py` → `WordCounter`, `wordlevel_pipeline` | Transcript text, resolved language | Word counts, entity table, noun sentiment table, graph HTML, word cloud `Figure` | CLI `-w/--words`, Streamlit "Word-level analysis" |
| Topic Modeling | `nextext/modules/topics.py` → `TopicModeling`, `topics_pipeline` | Transcript sentences, language, Ollama client | List of `(title, summary)` tuples | CLI `-tm/--topics`, Streamlit "Topic modelling" |
| Summarization | `nextext/pipeline.py` → `summarization_pipeline` + `OllamaPipeline` | Full transcript text | Summary string in Ollama output language | CLI `-sum/--summarize`, Streamlit "Summarisation" |
| Hate Speech Classification | `nextext/modules/hatespeech.py` → `HateSpeechDetector`, `hatespeech_pipeline` | Transcript sentences, Ollama client | Transcript `pd.DataFrame` with `hate_speech` column (0/1) | CLI `-tox/--toxicity`, Streamlit "Hate Speech" |
| File Export | `nextext/modules/processing.py` → `FileProcessor.write_file_output` | Any agent result | Files in `output/<input-file>/` (`.txt`, `.csv`, `.xlsx`, `.png`) | CLI workflow |
| Ollama Service | `nextext/modules/ollama_cfg.py` → `OllamaPipeline` | Prompt template + runtime options | Raw string from Ollama chat endpoint | Shared dependency for topic, summary, hate agents |

## Execution Flow

1. **Interface orchestrator** (CLI or Streamlit) collects user options and instantiates `transcription_pipeline`.
2. **Transcription agent** returns a `pd.DataFrame` and detected language code; the orchestration state is updated with the resolved language.
3. **Optional agents** (translation, word-level, topics, summary, hate speech) fire in the order shown in `nextext/app.py` or `nextext/cli.py`, each mutating or extending the transcript payload.
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
- **Responsibilities:** Detect the transcript language via `langdetect`, select the best MADLAD checkpoint (`nextext/utils/mappings/madlad_models.json`), and rewrite each sentence to the requested target.
- **Inputs:** Transcript `pd.DataFrame` (only the `text` column is used), target ISO 639-1 code, optional manual source (detected automatically otherwise).
- **Outputs:** In-place replacement of the `text` column; `Translator.src_lang` is populated for logging and downstream toggles.
- **Dependencies:** `transformers`, `torch`, `nltk` (sentence tokenization), `pyarabic` for Arabic segmentation; models load from cache first to support offline setups (`HF_HUB_OFFLINE=1`).
- **Operational notes:** Translation is skipped when the detected language already equals the target. Watch for the 256-token window in `_model_inference`; long passages are truncated unless chunked beforehand.

## Word Intelligence Agent

- **Key files:** `nextext/modules/words.py`, `wordlevel_pipeline()` (`nextext/pipeline.py`), spaCy mappings in `nextext/utils/mappings/spacy_models.json`.
- **Responsibilities:** Turn transcript text into linguistic diagnostics—top words, named entities, noun sentiment table, noun-verb-adjective network, and a matplotlib word cloud.
- **Inputs:** `pd.DataFrame` with `text`, resolved language code (source for `transcribe`, target for `translate`).
- **Outputs:** Tuple `(word_counts_df, entities_df, noun_sentiment_df, noun_graph_html_path, wordcloud_figure)`.
- **Dependencies:** `spacy`, `pyvis`, `matplotlib`, `camel_tools`, `arabic_reshaper`, `networkx`; fonts loaded via `nextext/utils/font_loader.py` to keep multilingual rendering stable.
- **Operational notes:** Call `text_to_doc()` then `lemmatize_doc()` before counting; spaCy models are downloaded on demand, so make sure the environment can write to the cache or pre-bundle them.

## Topic Modeling Agent

- **Key files:** `nextext/modules/topics.py`, `topics_pipeline()` (`nextext/pipeline.py`), prompts in `nextext/utils/prompts/topic_title.txt` and `topic_summary.txt`.
- **Responsibilities:** Sentence-tokenize the corpus, embed with a multilingual SentenceTransformer, cluster via UMAP + HDBSCAN, label each cluster, then ask Ollama for human-readable titles and summaries.
- **Inputs:** Transcript sentences (`list[str]`), language code (converted to language name through `pycountry`), `OllamaPipeline` instance to power zero-shot labeling.
- **Outputs:** List of `(title, summary)` tuples suitable for the UI table or CSV export; returns `None` when clustering fails.
- **Dependencies:** `bertopic`, `sentence-transformers`, `umap-learn`, `hdbscan`, `spacy`, `nltk`, `ollama`.
- **Operational notes:** Low-content files can produce `-1` topics; guard for this when consuming the results. Arabic uses a custom tokenizer when spaCy data is unavailable.

## Summarization Agent

- **Key files:** `summarization_pipeline()` (`nextext/pipeline.py`), prompt template `nextext/utils/prompts/summary.txt`.
- **Responsibilities:** Format the entire transcript as a prompt, apply the system instruction defined in `OllamaPipeline`, and return a concise summary limited to 15 sentences.
- **Inputs:** Concatenated transcript string, `OllamaPipeline`.
- **Outputs:** Summary string; orchestrators attach it to session state or export it via `FileProcessor`.
- **Dependencies:** Ollama server reachable via `OLLAMA_HOST` (defaults to `http://localhost:11434`); prompt language controlled by `OllamaPipeline.out_language`.
- **Operational notes:** The pipeline raises `ValueError` when text is empty; make sure optional agents check for data before calling.

## Hate Speech Classification Agent

- **Key files:** `nextext/modules/hatespeech.py`, prompt template `nextext/utils/prompts/hate.txt`, `hatespeech_pipeline()` (`nextext/pipeline.py`).
- **Responsibilities:** Format each transcript sentence with the hate-speech prompt, request a binary (0/1) label from Ollama, and append the results to the transcript DataFrame.
- **Inputs:** Transcript `pd.DataFrame`, `OllamaPipeline`.
- **Outputs:** Same `pd.DataFrame` with an added `hate_speech` column; values of `-1` indicate parsing problems and should be filtered upstream.
- **Dependencies:** `ollama`, prompt engineering defined in `hate.txt`.
- **Operational notes:** `_run_inference()` enforces deterministic sampling (`top_k=1`, `top_p=0`), so any non-binary response is logged and returns `-1`; monitor logs when tuning prompts.

## Ollama Service Agent

- **Key files:** `nextext/modules/ollama_cfg.py`, prompts directory `nextext/utils/prompts/`, model mapping `nextext/utils/mappings/ollama_models.json`.
- **Responsibilities:** Construct prompts, pick an Ollama model suited to the host hardware (CUDA, MPS, CPU), perform health checks (`/api/tags`), and expose `call_ollama_server()` to other agents.
- **Inputs:** Prompt keyword (`system`, `summary`, `topic_title`, `topic_summary`, `hate`), runtime options (context window, temperature, stop tokens).
- **Outputs:** Raw string response from `ollama.chat`; `sys_prompt` ensures outputs are emitted in the configured language.
- **Operational notes:** Set `OLLAMA_HOST` to point at the shared Ollama instance when running containers. If health checks fail the dependent agents are short-circuited with a `RuntimeError`.

## File Export Agent

- **Key files:** `nextext/modules/processing.py`.
- **Responsibilities:** Create `output/<input-file>/` directories and serialize any agent output (text, list, DataFrame, Matplotlib figure) to disk.
- **Inputs:** Label describing the payload, optional target language suffix.
- **Outputs:** `.txt`, `.csv`, `.xlsx`, or `.png` files depending on type; log statements confirm each save path.
- **Operational notes:** Only the CLI uses `FileProcessor`; the Streamlit UI instead keeps the artifacts in memory and displays/downloads them directly.

## Orchestrators (CLI & Streamlit)

- **Streamlit UI (`nextext/app.py`):** Stores user choices in `st.session_state["opts"]`, uploads files into a temporary path, and runs `_run_pipeline()`. Results are cached per session; UI tabs read from `st.session_state["result"]`.
- **CLI (`nextext/cli.py`):** `parse_arguments()` maps command-line flags to agent toggles. `main()` wires the pipelines sequentially and persists outputs via `FileProcessor`.
- **Shared behaviour:** Both orchestrators guard optional agents behind flags, update the resolved language after transcription, and instantiate a single `OllamaPipeline` per run to amortize health checks and prompt loading.

## Adding or Modifying Agents

1. **Create the module:** Place new agent code under `nextext/modules/` (or `nextext/pipeline.py` if it is a thin wrapper). Keep the public function signature narrow and return plain Python or pandas objects.
2. **Expose a pipeline hook:** Add a helper function in `nextext/pipeline.py` so both orchestrators can call the agent without duplicating logic.
3. **Wire toggles:** Update `nextext/app.py` (checkbox/radio button + session state) and `nextext/cli.py` (argument flag) so users can opt in/out.
4. **Document configuration:** Extend this file and, if needed, add prompt templates or mapping entries under `nextext/utils/`.
5. **Persist outputs:** Decide whether the result is UI-only or needs a saved artifact; if so, use `FileProcessor.write_file_output()` to follow the existing naming convention.

By keeping each agent isolated and documented here, Nextext can scale its audio-analysis capabilities without creating tight coupling between new features and the UI/CLI front ends.
